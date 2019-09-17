from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from att_speech.logger import DefaultTensorLogger

import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
import time

import pywrapfst as fst

from att_speech.modules.decoders.base_decoder import BaseDecoder
from att_speech.configuration import Globals
from att_speech.modules.common import SequenceWise
from att_speech.modules.ctc_losses import (
    ctc_loss, ctc_fst_loss, ctc_fst_global_loss,
    raw_generic_ctc, raw_generic_ctc_batch, get_normalized_acts)
from att_speech import utils, fst_utils


logger = DefaultTensorLogger()


class LutLinear(nn.Linear):
    """
    Look-up table for softmax (i.e. one prototype for each class).
    """
    def __init__(self, in_dim, num_symbols, ngram_to_class, bias=True,
                 tie_blanks=False, bias_only_for_dim=None):
        self.num_symbols = num_symbols
        if tie_blanks:
            tied_w_rows = torch.arange(ngram_to_class.size(0),
                                       dtype=torch.long)
            tied_w_rows[::num_symbols] = 0
            self.tied_w_rows = tied_w_rows
        else:
            self.tied_w_rows = None
        # Weight noise is set by a training hook
        # This assignment just says that we handle the situation
        self.weight_noise = 0.0
        super(LutLinear, self).__init__(
            in_dim, ngram_to_class.size(0), bias)

    def forward(self, input):
        w = self.weight
        if self.training:
            logger.log_scalar("ngram_linear_weight_norm",
                              torch.norm(w))
        if self.training and self.weight_noise > 0:
            noise = torch.randn_like(w)
            noise *= self.weight_noise
            print("Lut weight noise: %f " % (self.weight_noise))
            w = w + noise
        b = self.bias
        if self.tied_w_rows is not None:
            w = w[self.tied_w_rows]
            b = b[self.tied_w_rows]
        return F.linear(input, w, b)

    def extra_repr(self):
        return ('in_features={}, out_features={}, bias={}, tie_blanks={}, '
                'num_symbols={}'.format(
                    self.in_features, self.out_features,
                    self.bias is not None,
                    self.tied_w_rows is not None, self.num_symbols))


class GatedAct(nn.Module):
    def forward(self, x):
        g, x = torch.chunk(x, 2, dim=-1)
        return torch.sigmoid(g) * torch.tanh(x)


class NGramLinear(nn.Module):
    """
    Tied prototype mapping for softmax.

    args:
        - in_dim: input dimension
        - num_symbols: number of symbols (e.g. chars or phones)
        - ngram_to_class: array of size num_classes x ngram_order mapping
            each output class to a sequence of symbols
        - bias: whether to use a bias vector
        - bias_only_for_dim: if None, each ngram gets its own bias, else
            biases are added for the symbol in the given ngram location.
        - inner_dim: dimensionality of the inner layer
        - embedding_dim: dimensionality of the embedding
        - tied_embeddings: if sybol embeddings are tied across locations
        - embedding_combination_method: concat, sum or lstm
        - num_layers: number of layers of nonlinear embedding postprocessing
    """
    def __init__(self, in_dim, num_symbols, ngram_to_class, bias=True,
                 bias_only_for_dim=None,
                 inner_dim=None,
                 dropout=0.0,
                 embedding_dim=None, tied_embeddings=True,
                 embedding_combination_method='sum',
                 num_layers=0, weight_noise=0.0,
                 activation='relu'):
        super(NGramLinear, self).__init__()
        self.num_symbols = num_symbols
        self.ngram_to_class = ngram_to_class
        num_classes, ngram_order = self.ngram_to_class.size()
        self.in_dim = in_dim
        self.inner_dim = inner_dim or in_dim
        self.dropout = dropout

        embedding_multiplier = {
                'concat': ngram_order,
                'lstm': 1,
                'sum': 1
                }[embedding_combination_method]
        assert not (embedding_combination_method == 'lstm' and num_layers > 1)
        embedding_dim = embedding_dim or self.inner_dim // embedding_multiplier

        activation_class = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'gated': GatedAct}[activation]

        def layer_dim(lnum):
            if lnum == num_layers - 1:
                return self.in_dim
            ret_dim = self.inner_dim
            if activation == 'gated':
                ret_dim *= 2
            return ret_dim

        self.embedding_dim = embedding_dim
        self.tied_embeddings = tied_embeddings
        self.embedding_combination_method = embedding_combination_method
        self.num_layers = num_layers
        # Weight noise is set by a training hook
        # This assignment just says that we handle the situation
        self.weight_noise = 0.0

        if bias:
            self.bias_only_for_dim = bias_only_for_dim
            if bias_only_for_dim is None:
                self.bias = nn.Parameter(torch.Tensor(num_classes))
            else:
                self.bias = nn.Parameter(torch.Tensor(num_symbols, 1))
                self.ngram_to_bias = self.ngram_to_class[:, bias_only_for_dim]
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)

        if tied_embeddings:
            num_embeddings = self.num_symbols
        else:
            # This effectively lets each symbol-location pair to be a
            # different. Thus, the embeddings become wependent on location.
            num_embeddings = self.num_symbols * ngram_order
            shift = torch.arange(ngram_order, dtype=torch.long).view(1, -1) * \
                num_symbols
            if Globals.cuda:
                shift = shift.cuda()
            self.ngram_to_class = self.ngram_to_class + shift

        self.embedding = torch.nn.Embedding(num_embeddings,
                                            embedding_dim)

        if num_layers > 0:
            net_input_dim = embedding_multiplier * embedding_dim
            layers = [nn.Linear(net_input_dim, layer_dim(0))]
            if dropout:
                layers.append(nn.Dropout(dropout))
            for lnum in range(1, num_layers):
                layers.append(activation_class())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(layer_dim(lnum - 1),
                                        layer_dim(lnum)))
            self.weight_computer = nn.Sequential(*layers)
        elif self.embedding_combination_method == 'lstm':
            self.weight_computer_ = nn.LSTM(self.in_dim, self.in_dim,
                                            num_layers=1)
            self.weight_computer = lambda x: self.weight_computer_(x)[0][-1]
        else:
            self.weight_computer = lambda x: x

    def forward(self, input):
        embedded = self.embedding(self.ngram_to_class)
        if self.embedding_combination_method == 'concat':
            embedded = embedded.view(embedded.size(0), -1)
        elif self.embedding_combination_method == 'sum':
            embedded = embedded.sum(1)
        elif self.embedding_combination_method == 'lstm':
            embedded = embedded.transpose(0, 1)
        else:
            raise ValueError("Unknown embedding_combination_method")
        weight = self.weight_computer(embedded)
        if self.training:
            logger.log_scalar("ngram_linear_weight_norm",
                              torch.norm(weight))
        if self.training and self.weight_noise > 0:
            print("NoLut weight noise: %f " % (self.weight_noise))
            noise = torch.randn_like(weight)
            noise *= self.weight_noise
            weight = weight + noise
        if self.bias is not None:
            if self.bias_only_for_dim is None:
                bias = self.bias
            else:
                bias = F.embedding(self.ngram_to_bias, self.bias).view(-1)
        else:
            bias = None
        return F.linear(input, weight, bias)

    def extra_repr(self):
        return ('in_dim={}, num_symbols={}, ngram_to_class={}, bias={}, '
                'embedding_dim={}, tied_embeddings={}, '
                'embedding_combination_method={}, num_layers={}'.format(
                   self.in_dim, self.num_symbols,
                   self.ngram_to_class.size(), self.bias is not None,
                   self.embedding_dim, self.tied_embeddings,
                   self.embedding_combination_method, self.num_layers
                   ))


class CTCDecoderAdvanced(BaseDecoder):

    def __init__(self, sample_batch, num_classes, context_order=1,
                 normalize_by_dim=None, ctc_loss_fn='ctc_loss',
                 ctc_allow_nonblank_selfloops=True,
                 loop_using_symbol_repetitions=False,
                 embedder='LutLinear', embedder_kwargs={},
                 bigram_dovetail_decoder=False,
                 local_normalization=True,
                 fix_greedy_decoder=False,
                 **kwargs):
        super(CTCDecoderAdvanced, self).__init__(**kwargs)
        self.ctc_loss_fn = globals()[ctc_loss_fn]

        num_symbols, ngram_to_class, blanks = self.make_ngram_table(
            context_order, num_classes)
        assert self.num_symbols == num_symbols
        self.context_order = context_order
        self.normalize_by_dim = normalize_by_dim
        self.bigram_dovetail_decoder = bigram_dovetail_decoder
        self.blanks = blanks
        self.ctc_allow_nonblank_selfloops = ctc_allow_nonblank_selfloops
        self.loop_using_symbol_repetitions = loop_using_symbol_repetitions
        self.local_normalization = local_normalization
        self.fix_greedy_decoder = fix_greedy_decoder

        rnn_hidden_size = sample_batch["features"].size()[2]
        embedder = globals()[embedder]
        # Keep the sequential for compatiblity and easier initialization
        # from old checkpoints
        modules = []
        modules.append(
            embedder(rnn_hidden_size, num_symbols, ngram_to_class,
                     **embedder_kwargs))
        fully_connected = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

    def make_ngram_table(self, context_order, num_classes):
        num_symbols = int(num_classes**(1.0/context_order))
        assert num_symbols**context_order == num_classes
        ngram_to_class = []
        for i in range(num_classes):
            ngram_to_class.append(
                i // num_symbols**np.arange(context_order)[::-1] %
                num_symbols
            )
        ngram_to_class = torch.LongTensor(ngram_to_class)
        if Globals.cuda:
            ngram_to_class = ngram_to_class.cuda()
        blanks = [i for i in range(num_classes) if i % num_symbols == 0]
        return num_symbols, ngram_to_class, blanks

    def logits(self, encoded, encoded_lens=None, normalize_logits=True):
        logits = self.fc(encoded)
        if self.local_normalization:
            logits = get_normalized_acts(
                logits, encoded_lens, self.num_symbols, self.context_order,
                self.normalize_by_dim, normalize_logits)
        return logits

    def get_ctc_losses(self, logits, logit_lens, texts, text_lens,
                       other_data_in_batch):
        text_cat = torch.cat(
                [t[:l] for (t, l) in zip(texts, text_lens.data)])
        ctc_losses = self.ctc_loss_fn(
            logits, text_cat, logit_lens, text_lens, self.num_symbols,
            self.context_order, self.normalize_by_dim,
            self.ctc_allow_nonblank_selfloops,
            self.loop_using_symbol_repetitions,
            other_data_in_batch)
        return ctc_losses

    def forward(self, encoded, encoded_lens, texts, text_lens, spkids=None,
                **other_data_in_batch):
        unnormalised_logits = self.fc(encoded)
        ctc_loss = self.get_ctc_losses(
            unnormalised_logits, encoded_lens, texts, text_lens,
            other_data_in_batch).sum()
        loss_dict = {
            'ctc_loss': ctc_loss,
            }
        loss = ctc_loss
        loss_dict["loss"] = loss
        return loss_dict

    def decode(self, encoded, encoded_lens, texts=None, text_lens=None,
               return_texts_and_generated_loss=False,
               return_logits_text_diff=False, spkids=None,
               **other_data_in_batch):
        logits = self.logits(encoded, encoded_lens)
        ctc_loss = None
        if texts is not None and text_lens is not None:
            ctc_text_losses = self.get_ctc_losses(
                logits, encoded_lens, texts, text_lens,
                other_data_in_batch)
            ctc_loss = ctc_text_losses.sum()

        logits_t = logits.transpose(0, 1).cpu()
        if self.bigram_dovetail_decoder:
            assert self.context_order == 2
            path = torch.tensor(logits_t).detach()
            bsz, time_, num_biphones = path.size()
            maxes = torch.zeros((bsz, time_))
            for step in range(1, time_):
                for b in range(bsz):
                    rights = torch.tensor(
                        [torch.max(path[b, step-1, i::self.num_symbols])
                         for i in range(self.num_symbols)])
                    rights = rights.reshape(self.num_symbols, 1).repeat(
                        1, self.num_symbols).reshape(num_biphones)
                    path[b, step] += rights
            # now go back
            for b in range(bsz):
                maxes[b, -1] = torch.argmax(path[b, -1])
            rights = torch.zeros(self.num_symbols, self.num_symbols)
            for i in range(self.num_symbols):
                rights[i][i] = 1
            rights = rights.repeat(1, self.num_symbols)
            for step in range(time_-1, 0, -1):
                for b in range(bsz):
                    left = int(maxes.data[b, step].item() // self.num_symbols)
                    path[b, step-1] += rights[left]
                    maxes[b, step-1] = torch.argmax(path[b, step-1])
        else:
            _, maxes = torch.max(logits_t, 2)
        decoded = self.process_sequences(maxes, encoded_lens)
        ret = {'decoded': decoded, 'decoded_frames': maxes, 'logits': logits}
        if ctc_loss is not None:
            loss_dict = {}
            loss_dict['ctc_loss'] = ctc_loss

            loss = ctc_loss
            loss_dict["loss"] = loss
            ret['loss'] = loss_dict

        if return_texts_and_generated_loss:
            decoded_lens = torch.IntTensor([len(x) for x in decoded])
            ctc_generated_losses = self.get_ctc_losses(
                logits, encoded_lens, decoded, decoded_lens, {})
            ret['text_loss'] = ctc_text_losses.tolist()
            ret['generated_loss'] = ctc_generated_losses.tolist()

        if return_logits_text_diff:
            ret['logits_text_diff'] = (encoded_lens - text_lens).tolist()
        return ret

    def process_sequences(self, logits, logits_lens):
        return [self.process_sequence(logits[i, :], logits_lens[i])
                for i in range(len(logits_lens))]

    def process_sequence(self, logits, logits_len):
        if self.fix_greedy_decoder:
            return [s % self.num_symbols for s in
                    remove_repetitions_blanks(
                        logits[:logits_len.item()].detach().cpu().numpy(),
                        self.blanks)]

        ret = []
        for i, char in enumerate(logits[:logits_len.item()]):
            if char not in self.blanks and (i != 0 or char != logits[i-1]):
                if not ret or (ret[-1] % self.num_symbols !=
                               char % self.num_symbols):
                    ret.append(char)
        return ret


class FSTDecoder(BaseDecoder):
    def __init__(self, sample_batch, num_classes,
                 graph_generator, normalize_by_dim=None,
                 numerator_red='logsumexp',
                 denominator_red='logsumexp',
                 embedder='LutLinear', embedder_kwargs={},
                 **kwargs):
        super(FSTDecoder, self).__init__(**kwargs)

        self.graph_generator = utils.contruct_from_kwargs(
                graph_generator, 'att_speech.fst_utils',
                {'num_classes': num_classes,
                 'num_symbols': self.num_symbols})

        self.context_order = self.graph_generator.context_order
        self.normalize_by_dim = normalize_by_dim
        self.numerator_red = numerator_red
        self.denominator_red = denominator_red
        self._verify = False

        # None means no normalization
        # 0 means softmax over all symbols
        # >0 means normlize within contexts
        if self.normalize_by_dim not in [None, 0]:
            assert (self.graph_generator.num_classes ==
                    self.graph_generator.num_symbols ** self.context_order)

        # An fst used for decoding most probable state sequences
        self.dec_fst = fst.arcmap(
            self.graph_generator.decoding_fst, 0, 'to_standard', 0
            ).arcsort('ilabel')

        rnn_hidden_size = sample_batch["features"].size()[2]
        embedder = globals()[embedder]
        # Keep the sequential for compatiblity and easier initialization
        # from old checkpoints
        ngram_to_class = self.graph_generator.ngram_to_class
        if Globals.cuda:
            ngram_to_class = ngram_to_class.cuda()
        modules = []
        modules.append(
            embedder(rnn_hidden_size,
                     self.graph_generator.num_symbols,
                     ngram_to_class,
                     **embedder_kwargs))
        fully_connected = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

    def logits(self, encoded, encoded_lens=None, extra_ret=None):
        logits = self.fc(encoded)
        if extra_ret is not None:
            extra_ret['unnormed_logits'] = logits
        if self.normalize_by_dim is not None:
            logits = get_normalized_acts(
                logits, encoded_lens, self.num_symbols, self.context_order,
                self.normalize_by_dim, normalize_logits=True)
        return logits

    def get_fst_loss(self, logits, encoded_lens, texts, text_lens,
                     other_data_in_batch):
        tg = time.time()
        if other_data_in_batch and 'graph_matrices' in other_data_in_batch:
            numerator_matrices = [gm.to(logits.device) for gm in
                                  other_data_in_batch['graph_matrices']]
            if not self._verify:
                numerator_matrices2 = \
                    self.graph_generator.get_training_matrices_batch(
                        texts, text_lens, logits.device)
                assert max([torch.abs(m1 - m2).max().item()
                            for m1, m2 in zip(numerator_matrices,
                                              numerator_matrices2)]
                           ) < 1e-10
                self._verify = True
        else:
            numerator_matrices = self.graph_generator.get_training_matrices_batch(
                texts, text_lens, logits.device)

        denominator_matrices = self.graph_generator.get_decoding_matrices(
            logits.device)

        tnorm = time.time()
#         logitso = logits.detach()

        mask = (torch.arange(logits.size(0), dtype=encoded_lens.dtype,
                             device=logits.device).unsqueeze(1) <
                encoded_lens.to(logits.device)).unsqueeze(-1).float()
        logits_max = logits.max(-1, keepdim=True)[0].detach()
        logits_sum = (logits_max * mask.float()).sum(0).squeeze(-1)
        logits = logits - logits_max

        ts = time.time()
        neg_inf = self.graph_generator.nc_weight
        numerator_loss = -fst_utils.path_reduction(
            logits, encoded_lens, numerator_matrices,
            red_kind=self.numerator_red, neg_inf=neg_inf)

#         numerator_losso = -fst_utils.path_reduction(
#             logitso, encoded_lens, numerator_matrices,
#             red_kind=self.numerator_red, neg_inf=neg_inf)

        tn = time.time()
        if self.denominator_red != 'none':
            denominator_loss = -fst_utils.path_reduction(
                logits, encoded_lens, denominator_matrices,
                red_kind=self.denominator_red, neg_inf=neg_inf)
#             denominator_losso = -fst_utils.path_reduction(
#                 logitso, encoded_lens, denominator_matrices,
#                 red_kind=self.denominator_red, neg_inf=neg_inf)
        else:
            denominator_loss = logits_sum
#             denominator_losso = torch.zeros_like(logits_sum)

        print("global loss: [time: gprep: %f norm %f num %f, den  %f;"
              "loss: num %g, den %g, com %g]" % (
                  tnorm - tg, ts - tnorm, tn - ts, time.time() - tn,
                  numerator_loss.sum().item(),
                  denominator_loss.sum().item(),
                  -logits_sum.sum().item()))
#         print("num", numerator_loss - logits_sum, numerator_losso)
#         print("den", denominator_loss - logits_sum, denominator_losso)

        return numerator_loss - denominator_loss

    def forward(self, encoded, encoded_lens, texts, text_lens, spkids=None,
                **other_data_in_batch):
        extra_ret = {}
        logits = self.logits(encoded, encoded_lens, extra_ret=extra_ret)

        fst_losses = self.get_fst_loss(
            logits, encoded_lens, texts, text_lens, other_data_in_batch
            )
        fst_loss = fst_losses.sum()

        loss_dict = {
            'fst_loss': fst_loss,
            }
        loss = fst_loss
        loss_dict["loss"] = loss
        return loss_dict

    def decode(self, encoded, encoded_lens, texts=None, text_lens=None,
               return_texts_and_generated_loss=False,
               return_logits_text_diff=False, spkids=None,
               **other_data_in_batch):
        logits = self.logits(encoded, encoded_lens)

        denominator_matrices = self.graph_generator.get_decoding_matrices(
            logits.device)

        # The gradient wrt Viterbi gives the symbols on the shortest path.
        with torch.enable_grad():
            logits = logits.detach().requires_grad_()
            loss = -fst_utils.path_reduction(
                logits, encoded_lens, denominator_matrices,
                red_kind='viterbi', neg_inf=self.graph_generator.nc_weight
                ).sum()
            loss.backward()

        selidx = logits.grad.min(-1)[1].cpu()
        decoded_texts = []
        for i in range(logits.size(1)):
            # add 1 because our FST wants symbols from range 1..Num_Symbols
            idx = selidx[:encoded_lens[i], i] + 1
            decoded_fst = fst.shortestpath(
                fst.compose(
                    fst_utils.build_chain_fst(idx, arc_type='standard'),
                    self.dec_fst)
                )
            decoded_text = []
            n = decoded_fst.start()
            while decoded_fst.num_arcs(n) != 0:
                a, = decoded_fst.arcs(n)
                n = a.nextstate
                if a.olabel > 0:
                    decoded_text.append(a.olabel)
            decoded_texts.append(decoded_text)

        ret = {
            'decoded': decoded_texts,
            # 'decoded_frames': selidx,
            'logits': logits}

        if texts is not None and text_lens is not None:
            fst_text_losses = self.get_fst_loss(
                logits, encoded_lens, texts, text_lens, other_data_in_batch)
            fst_text_loss = fst_text_losses.sum()
            ret['loss'] = dict(fst_loss=fst_text_loss, loss=fst_text_loss)

        if return_texts_and_generated_loss:
            decoded_lens = torch.IntTensor([len(x) for x in decoded_texts])
            fst_generated_losses = self.get_fst_loss(
                logits, encoded_lens, decoded_texts, decoded_lens, None)
            ret['text_loss'] = fst_text_losses.tolist()
            ret['generated_loss'] = fst_generated_losses.tolist()

        if return_logits_text_diff:
            ret['logits_text_diff'] = (encoded_lens - text_lens).tolist()
        return ret

