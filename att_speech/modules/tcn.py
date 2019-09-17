from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import numpy as np


import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from collections import OrderedDict

from att_speech.logger import DefaultTensorLogger
from att_speech.modules.common import SequenceWise
from att_speech.modules.beam_search import BeamSearch, BeamSearchLM, GraphSearch, RescoreSearchLM
from att_speech.utils import get_mask
from att_speech.configuration import Globals
from att_speech import fst_utils

import pywrapfst as fst
import time


logger = DefaultTensorLogger()

# https://arxiv.org/pdf/1803.01271.pdf
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        '''
        :param x:
        in shape batch size x hidden size x seq len
         '''
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2, n_layers=2):
        super(TemporalBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            n_in = n_inputs if i == 0 else n_outputs
            conv = torch.nn.utils.weight_norm(nn.Conv1d(n_in, n_outputs,
                                                        kernel_size,
                                                        stride=stride,
                                                        padding=padding,
                                                        dilation=dilation))
            chomp = Chomp1d(padding)
            relu = nn.ReLU()
            drop = nn.Dropout2d(dropout)
            layers += [
                    ('conv' + str(i), conv),
                    ('chomp' + str(i), chomp),
                    ('relu' + str(i), relu),
                    ('drop' + str(i), drop)]
        self.net = nn.Sequential(OrderedDict(layers))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
            if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for name, layer in self.net.named_children():
            if name.startswith('conv'):
                layer.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        '''
        :param x:
        in shape batch size x hidden size x seq_len
        '''
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, dilation_sizes, kernel_size=2,
                 dropout=0.2, layers_per_block=2):
        super(TCN, self).__init__()
        self.eff_history = sum(dilation_sizes) * (kernel_size - 1) + 1
        layers = []
        num_levels = len(num_channels)
        if len(dilation_sizes) != num_levels:
            raise ValueError(('num_channels and dilations_sizes lengths '
                              'must be equal (number of blocks)'))
        for i in range(num_levels):
            dilation_size = dilation_sizes[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, dropout=dropout,
                n_layers=layers_per_block)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        '''
        x is transformed from seq len x batch size x hidden size as used in
        decoder into batch_size x hidden_size x seq_len
        as used by conv1d and tcn
        '''
        return self.network(x.permute(1, 2, 0)).permute(2, 0, 1)


class LocalAttention(nn.Module):
    def __init__(self, encoded_size, lm_state_size, hidden_size,
                 kernel_size=32, temperature=1.0, force_forward=None,
                 learnable_init=True, **kwargs):
        super(LocalAttention, self).__init__(**kwargs)
        self.encoded_size = encoded_size
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.force_forward = force_forward
        # TODO weights and biases initialization
        self.encoded_to_hidden = nn.Linear(encoded_size, hidden_size)
        # self.lm_state_to_hidden = nn.Linear(lm_state_size, hidden_size,
        #                                     bias=False)
        self.hidden_to_score = nn.Linear(hidden_size, 1)
        self.lm_to_kernel = nn.Linear(lm_state_size, kernel_size * hidden_size)
        # self.lm_to_kernel = nn.Linear(lm_state_size, kernel_size)
        self.lm_to_global = nn.Linear(lm_state_size, hidden_size)
        # Zero initialization means that initially we will average all
        self.hidden_to_score.weight.data.zero_()
        self.encoded_to_init_weights = nn.Linear(encoded_size, 1)
        self.learnable_init = learnable_init

    def init_attention(self, encoded, encoded_lens):
        '''
        :param encoded:
            encoded sequence with shape (T, B, H)
        :param encoded_lens:
            lengths of encoded sequences of shape (B,)
        :return
            att_state: pair of encoded's contribution to the attention
            computation of shape (B, H) and attention masks of shape (T, B)
        '''
        encoded_contribution = self.encoded_to_hidden(encoded)
        mask = get_mask(encoded_lens, encoded.size(0), batch_first=False)
        mask = (mask - 1.0) * 1e5
        if encoded_contribution.is_cuda:
            mask = mask.cuda()
        scores = self.encoded_to_init_weights(encoded).squeeze(2)
        scores += mask
        if self.learnable_init:
            att_weights = F.softmax(scores, 0)
        else:
            att_weights = torch.zeros_like(scores)
            att_weights[0, :] = 1  # attention on first frame
        return (encoded_contribution, mask), att_weights

    def recompute_forward_mask(self, prev_att_weights, mask):
        '''
        :param prev_att_weight:
            previous attention weights, in shape (BS, B, T)
        :param mask:
            default mask, of shape (T, B)
        :return
            new mask with forced alignment
        '''
        att_max, max_indicies = torch.max(prev_att_weights, 2)
        max_indicies = max_indicies.view(-1)  # batch_size x beam_size
        mask = mask.clone()
        for j in range(max_indicies.shape[0]):
            if att_max[0, j].item() < 0.1:  # diffused attention, don't mask!
                # print("Diffused att")
                continue
            ind = max_indicies[j]
            left = ind + self.force_forward[0]
            right = ind + self.force_forward[1]
            if left > 0:
                mask[:left, j] -= 1e5
            if right < mask.shape[0]:
                mask[right:, j] -= 1e5
        return mask

    def forward(self, att_state, lm_state, prev_att_weights):
        '''
        :param att_state:
            computed by init_attention or forward
        :param lm_state:
            previous output of the language model, in shape
            (B,H)
        :param prev_att_weights:
            previous attention weights, in shape (T, B)
        :return
            pair of attn_state and attention energies in shape (T, B)
        '''
        encoded_contribution, mask = att_state
        encoded_len = prev_att_weights.size(0)
        # Part 1: local attention.
        # Computed by moving previous attention with a 1D convolution
        # using kernel computed from lm_state.
        # TODO consider encoded also contributing to kernel, e.g. to capture
        # avarage pace of speech
        kernel = self.lm_to_kernel(lm_state)
        bs = kernel.size(0)
        # [out_channels = B * H, in_channels / groups = 1, kernel_size]
        kernel = kernel.view(-1, 1, self.kernel_size)
        prev_att_weights = prev_att_weights.t().unsqueeze(0)
        pad = self.kernel_size - 1
        local_hidden = F.conv1d(prev_att_weights, kernel,
                                padding=pad, groups=bs)[:, :, :-pad]
        local_hidden = local_hidden.transpose(0, 2) \
            .view_as(encoded_contribution)
        # local_hidden = local_hidden.transpose(0, 2)
        # Part 2: match lm_state with encoded globally.
        global_hidden = self.lm_to_global(lm_state).unsqueeze(0)
        # Part 3: combine with bias from encoded.
        hidden = encoded_contribution + local_hidden + global_hidden
        scores = self.hidden_to_score(torch.tanh(hidden)).squeeze(2)
        scores *= self.temperature
        if self.force_forward:
            mask = self.recompute_forward_mask(prev_att_weights, mask)
        scores += mask
        att_weights = F.softmax(scores, 0)
        return att_state, att_weights


class AttentionDecoderTCN(nn.Module):
    def __init__(self, sample_batch, num_classes,
                 tcn_hidden_size, att_hidden_size, dropout_p,
                 learnable_initial_attention=True, label_smoothing=True,
                 kernel_size=3, dilation_sizes=[1, 2, 4],
                 coverage_tau=0.5, coverage_weight=0,
                 beam_size=1, length_normalization=0.0,
                 att_force_forward=None, vocabulary=None,
                 branching_threshold=0.0, lm_file=None, lm_weight=1.0,
                 attention_temperature=1.0, tcn_layers_per_block=2,
                 min_attention_pos=0.5, keep_eos_score=False,
                 use_graph_search=False, graph_search_history_len=-1,
                 graph_search_merge_threshold=0.8, **kwargs):
        super(AttentionDecoderTCN, self).__init__(**kwargs)

        self.rescore = None
        # Define parameters
        self.coverage_tau = coverage_tau
        self.coverage_weight = coverage_weight
        self.encoded_size = sample_batch["features"].size()[2]
        self.tcn_hidden_size = tcn_hidden_size
        self.att_hidden_size = att_hidden_size
        self.num_classes = num_classes + 1  # adding EOS
        self.EOS = num_classes
        self.vocabulary = vocabulary
        self.use_graph_search = use_graph_search
        self.min_attention_pos = min_attention_pos
        self.keep_eos_score = keep_eos_score
        self.graph_search_history_len = graph_search_history_len
        self.graph_search_merge_threshold = graph_search_merge_threshold

        # Define layers
        self.embedding = nn.Embedding(self.num_classes, tcn_hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = LocalAttention(self.encoded_size, tcn_hidden_size,
                                   att_hidden_size,
                                   temperature=attention_temperature,
                                   learnable_init=learnable_initial_attention,
                                   force_forward=att_force_forward)
        self.tcn = TCN(tcn_hidden_size,
                       [tcn_hidden_size] * len(dilation_sizes),
                       dilation_sizes=dilation_sizes,
                       kernel_size=kernel_size,
                       dropout=dropout_p,
                       layers_per_block=tcn_layers_per_block)
        out_size = 256
        self.combined_to_output = nn.Sequential(
                nn.Linear(self.tcn_hidden_size + self.encoded_size, out_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(out_size, out_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_p))
        self.output_to_logits = nn.Linear(out_size, self.num_classes)
        self.beam_size = beam_size
        self.length_normalization = length_normalization
        self.branching_threshold = branching_threshold
        self.TRANSCRIPTION_LEN_GUARD = 250
        self.lm_weight = lm_weight
        self.label_smoothing = label_smoothing
        if lm_file:
            self.lm = fst.Fst.read(lm_file)
            if not self.lm.properties(fst.I_LABEL_SORTED,
                                      fst.I_LABEL_SORTED):
                self.lm.arcsort()
                assert self.lm.properties(fst.I_LABEL_SORTED,
                                          fst.I_LABEL_SORTED)
            assert self.vocabulary is not None
        else:
            self.lm = None
        self.alphabet_mapping = self.create_alphabet_mapping()
        self.debug = False

    def create_alphabet_mapping(self):
        if self.lm is None:
            return
        # Mapping from model_char -> lm_char
        # lm: eps, spc, pad, unk, E, T, A...
        # model: pad, unk, spc, E, T, A, ..., EOS
        lm_symbols = list(self.lm.input_symbols())
        lm_ids, lm_syms = zip(*lm_symbols)
        default_id = lm_ids[lm_syms.index('<spc>')]

        model_symbols = self.vocabulary

        mapping = []
        for s in model_symbols + ['<eos>']:
            if s == ' ':
                s = '<spc>'
            try:
                idx = lm_syms.index(s)
            except ValueError:
                idx = default_id
            mapping += [lm_ids[idx]]

        return mapping

    def att_median(self, att_weights):
        cum_att_weights = torch.cumsum(att_weights, 0)
        att_median = (cum_att_weights > 0.5).argmax(0)
        return att_median

    def hash_dec(self, decoded):
        if self.graph_search_history_len >= 0:
            hs = self.graph_search_history_len
        else:
            hs = self.tcn.eff_history

        if hs == 0:
            return 0
        else:
            new_decoded = [-1] * (hs - len(decoded)) + decoded[-hs:].tolist()
            return hash(tuple(new_decoded))

    def att_prod(self, x, y):
        # print (torch.stack([x,y], dim = 1))
        return torch.sum(torch.min(x, y))

    def is_prefix(self, l1, l2):
        if len(l1) > len(l2):
            return False
        l2 = l2[:len(l1)]
        return all(l1 == l2)

    def forward(self, encoded, encoded_lens, texts, text_lens,
                return_att_weights=False, **kwargs):
        '''
        encoded -> max_encoded_len x bs x encoded_size
        texts -> bs x max_text_len
        '''
        # Init recurrence
        bs = texts.size(0)
        att_state, att_weights = self.attn.init_attention(encoded,
                                                          encoded_lens)
        texts = torch.cat((texts, (torch.zeros(bs, 1).int())), dim=1)
        for bs_it in range(bs):
            texts[bs_it, text_lens[bs_it]] = self.EOS
        max_text_len = texts.size()[1]
        if Globals.cuda:
            texts = texts.cuda()
        texts = texts.long()
        embedded = self.embedding(texts.t())
        # Now embedded are L x B x D
        all_att_weights = []
        outputs = []
        lm_outputs = self.tcn(
            torch.cat((torch.zeros(1, embedded.size(1), embedded.size(2))
                      .type_as(embedded), embedded[:-1])))
        for lm_output in lm_outputs:
            att_state, att_weights = \
                self.attn(att_state, lm_output, att_weights)
            # Att_weights are L x B
            all_att_weights.append(att_weights)

            context = (att_weights.unsqueeze(2) * encoded).sum(0)
            combined = torch.cat((lm_output, context), 1)
            output = self.combined_to_output(combined)
            outputs.append(output.unsqueeze(0))
        outputs = torch.cat(outputs)
        # TODO try this instead on GPU with more memory
        # all_att_weights = torch.cat(all_att_weights)
        # contexts = (all_att_weights.unsqueeze(3)
        #             * encoded.unsqueeze(0)).sum(1)
        # combined = torch.cat((lm_outputs, contexts), 2)
        # outputs = self.combined_to_output(combined)
        try:
            if logger.is_currently_logging():
                att_weights = torch.cat([aw[:, :1] for aw in all_att_weights],
                                        1).data.cpu().numpy()
                att_weights = np.tile(att_weights[:, :, None], (1, 1, 3))
                logger.log_image('attention_weigths', att_weights)
        except Exception as e:
            pass

        logits = self.output_to_logits(outputs)  # len x bs x feat
        logits = logits.permute(1, 0, 2).contiguous()

        # label smoothing
        targets = torch.zeros(
            bs, self.num_classes, texts.size(1)).to(logits.device)
        targets.scatter_(1, texts.unsqueeze(1), 1)
        if self.label_smoothing:
            smooth_kernel = (torch.Tensor([0.005, 0.02, 0.95, 0.02, 0.005])
                             .unsqueeze(0).unsqueeze(0).to(targets.device))
            targets = (F.conv1d(targets.view(
                                    bs*self.num_classes, 1, max_text_len),
                                smooth_kernel, padding=2)
                       .view(bs, self.num_classes, max_text_len)
                       .transpose(1, 2))
        else:
            targets = targets.transpose(1, 2)

        targets /= targets.sum(2).unsqueeze(2)
        targets[:, :, 0] = 0  # ignore index 0
        loss = (-(F.log_softmax(logits, 2) * targets).sum(2).mean()
                / targets.sum(2).mean())
        # loss = F.cross_entropy(logits.view(bs * max_text_len, -1),
        #                        texts.view(bs * max_text_len),
        #                        ignore_index = 0)
        predictions = torch.argmax(logits, dim=2)
        predictions = torch.where(texts == 0, texts, predictions)
        acc = (((predictions == texts).double() - (texts == 0).double()).mean()
               * (torch.ones_like(texts).sum().item()
               / texts.nonzero().size(0)))
        ret = {'loss': loss, 'acc': acc, 'logits': logits}
        if return_att_weights:
            ret['attweights'] = all_att_weights
        return ret

    def enc_initial_state(self, encoded, encoded_lens,
                          beam_size, batch_size):
        max_encoded_len = encoded.size()[0]

        inputs = torch.zeros(
                    self.tcn.eff_history, batch_size, self.tcn_hidden_size,
                    device=encoded.device)
        inputs = inputs.unsqueeze(2).repeat(1, 1, beam_size, 1) \
            .view(self.tcn.eff_history, batch_size * beam_size, -1)

        encoded = encoded.unsqueeze(2).repeat(1, 1, beam_size, 1) \
            .view(max_encoded_len, batch_size * beam_size, -1)
        extended_encoded_lens = encoded_lens.clone().unsqueeze(1) \
            .repeat(1, self.beam_size).view(-1)

        att_state, att_weights = \
            self.attn.init_attention(encoded, extended_encoded_lens)

        return {'inputs': inputs,
                'encoded': encoded,
                'att_state': att_state,
                'att_weights': att_weights}

    def enc_step(self, inputs, encoded, att_state, att_weights):
        lm_output = self.tcn(inputs)[-1]
        att_state, att_weights = self.attn(att_state, lm_output, att_weights)
        context = (att_weights.unsqueeze(2) * encoded).sum(0)
        combined = torch.cat((lm_output, context), 1).unsqueeze(0)
        output = self.combined_to_output(combined)
        logits = self.output_to_logits(output)
        return logits, {'encoded': encoded,
                        'att_state': att_state,
                        'att_weights': att_weights}

    def decode(self, encoded, encoded_lens, texts=None,
               text_lens=None, return_attention=False,
               print_debug=False, **kwargs):
        """
        encoded      -> max_len x bs x encoded_size
        encoded_lens -> bs
        texts        -> bs x max_text_len
        text_len     -> bs
        """
        if self.debug:
            import ptvsd
            ptvsd.enable_attach(
                address=('localhost', 5678), redirect_output=True)
            print('Wait for attach')
            ptvsd.wait_for_attach()
            print('Attached')

        batch_size = encoded.size()[1]
        beam_size = self.beam_size

        if self.lm:
            kes = self.keep_eos_score
            if self.rescore:
                beam_search = RescoreSearchLM(
                    self.rescore,
                    self.lm, self.lm_weight, self.alphabet_mapping,
                    self.min_attention_pos, self.coverage_tau,
                    self.coverage_weight,
                    batch_size, beam_size, encoded.device,
                    self.num_classes,
                    self.length_normalization,
                    keep_eos_score=kes)
            elif self.use_graph_search:
                beam_search = GraphSearch(
                    self.hash_dec, self.graph_search_merge_threshold,
                    self.lm, self.lm_weight, self.alphabet_mapping,
                    self.min_attention_pos, self.coverage_tau,
                    self.coverage_weight,
                    batch_size, beam_size, encoded.device,
                    self.num_classes,
                    self.length_normalization,
                    keep_eos_score=kes)
            else:
                beam_search = BeamSearchLM(
                    self.lm, self.lm_weight, self.alphabet_mapping,
                    self.min_attention_pos, self.coverage_tau,
                    self.coverage_weight,
                    batch_size, beam_size, encoded.device,
                    self.num_classes,
                    self.length_normalization,
                    keep_eos_score=kes)
        else:
            beam_search = BeamSearch(
                batch_size, beam_size, encoded.device,
                self.num_classes,
                self.length_normalization)
        beam_search.print_debug = print_debug

        enc_state = self.enc_initial_state(encoded, encoded_lens,
                                           beam_size, batch_size)

        all_att_weights = []
        if return_attention:
            all_logits = []

        if return_attention:
            all_att_weights += [enc_state['att_weights'].detach()]

        for iteration_counter in range(self.TRANSCRIPTION_LEN_GUARD):
            prev_inputs = enc_state['inputs']
            logits, enc_state = self.enc_step(**enc_state)
            if return_attention:
                all_logits += [logits.detach()]
                all_att_weights += [enc_state['att_weights'].detach()]

            new_input, state_mapping = beam_search.step(
                logits, att_weights=enc_state['att_weights'])

            # if iteration_counter == stop_check:
            enc_state['att_weights'
                      ] = enc_state['att_weights'][:, state_mapping]
            prev_inputs = prev_inputs[:, state_mapping]
            enc_state['inputs'] = torch.cat(
                (prev_inputs[1:], self.embedding(new_input).unsqueeze(0)))

#             if print_debug:
#                 for enc in beam_search.estimations:
#                     print(beam_search.to_text(enc))
#                     print(enc)
#                 print('---')

            if beam_search.has_finished():
                # print('Finished in {} steps'.format(iteration_counter))
                break

        results = {
            'decoded': beam_search.best_finished,
            'decoded_scores': beam_search.best_finished_scores_elements,
            'loss': torch.Tensor(beam_search.best_finished_scores).mean()
        }

        if return_attention:
            results['attweights'] = all_att_weights
            results['logits'] = all_logits

        results['coverage'] = beam_search.coverage
        results['graph'] = beam_search.get_graph()
        results['beam_search'] = beam_search

        return results

    def single_step(self,
                    word_input,
                    last_hidden,
                    encoder_outputs,
                    encoded_lens,
                    precomputed_V_enc_out):
        pass
