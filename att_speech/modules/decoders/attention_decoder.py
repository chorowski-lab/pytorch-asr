
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F

from att_speech.logger import DefaultTensorLogger
from att_speech.modules.common import SequenceWise
from att_speech.utils import get_mask
from att_speech.configuration import Globals

from att_speech.modules.beam_search import BeamSearch, BeamSearchLM, GraphSearch

import pywrapfst as fst

logger = DefaultTensorLogger()


class Attention(nn.Module):
    def __init__(self, encoded_size, rec_state_size, hidden_size,
                 force_forward=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.encoded_to_hidden = nn.Linear(encoded_size, hidden_size)
        self.rec_state_to_hidden = nn.Linear(rec_state_size, hidden_size,
                                             bias=False)
        self.hidden_to_score = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        # Zero initialization means that initially we will average all
        self.hidden_to_score.weight.data.zero_()
        self.force_forward = force_forward

    def init_attention(self, encoded, encoded_lens):
        '''
        :param encoded:
            encoded sequence with shape (T, B, H)
        :param encoded_lens:
            lengths of encoded sequences of shape (B,)
        :return
            att_state: pair of encoded's contribution to the attention
            computation and attention masks
        '''
        encoded_contribution = self.encoded_to_hidden(encoded)
        mask = get_mask(encoded_lens, encoded.size(0), batch_first=False)
        mask = (mask - 1.0) * 1e5
        if encoded_contribution.is_cuda:
            mask = mask.cuda()

        t, bs, encs = encoded.size()

        att_weights = torch.zeros(
            (t, bs), device=encoded.device)
        att_weights[0, :] = 1  # attention on first frame

        return (encoded_contribution, mask), att_weights

    def recompute_forward_mask(self, prev_att_weights, mask):
        '''
        :param prev_att_weight:
            previous attention weights, in shape (T, B)
        :param mask:
            default mask, of shape (T, B)
        :return
            new mask with forced alignment
        '''
        # print(prev_att_weights.shape)
        att_max, max_indicies = torch.max(prev_att_weights, 0)
        max_indicies = max_indicies.view(-1)  # batch_size x beam_size
        mask = mask.clone()
        # print(mask.shape)
        for j in range(max_indicies.shape[0]):
            if att_max[j].item() < 0.1:  # diffused attention, don't mask!
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

    def forward(self, att_state, rnn_state, prev_att_weights):
        '''
        :param att_state:
            computed by init_attention or forward
        :param rnn_state:
            previous hidden state of the decoder, in shape
            (layers*directions,B,H)
        :param prev_att_weight:
            previous attention weights, in shape (T, B)
        :return
            pair of attn_state and attention energies in shape (T, B)
        '''
        encoded_contribution, mask = att_state
        rec_state_contribution = self.rec_state_to_hidden(rnn_state)
        rec_state_contribution = torch.unsqueeze(rec_state_contribution, 0)
        hidden = encoded_contribution + rec_state_contribution
        scores = self.hidden_to_score(F.tanh(hidden)).squeeze(2)
        if self.force_forward:
            mask = self.recompute_forward_mask(prev_att_weights, mask)
        scores += mask
        att_weights = F.softmax(scores, 0)
        return att_state, att_weights


class AttentionDecoderRNN(nn.Module):
    def __init__(self, sample_batch, num_classes, n_layers, hidden_size,
                 dropout_p, lm_file=None, lm_weight=1.0, min_attention_pos=0.3,
                 coverage_tau=0.1, coverage_weight=0.5, beam_size=1,
                 att_force_forward=None,
                 length_normalization=1.2, keep_eos_score=False,
                 use_graph_search=False, vocabulary=None, **kwargs):
        super(AttentionDecoderRNN, self).__init__(**kwargs)

        # Define parameters
        self.encoded_size = sample_batch["features"].size()[2]
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes + 1  # adding EOS
        self.EOS = num_classes

        # Define layers
        self.embedding = nn.Embedding(self.num_classes, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attention(self.encoded_size, hidden_size, hidden_size,
                              att_force_forward)

        self.rnn = nn.GRU(hidden_size + self.encoded_size,
                          hidden_size,
                          n_layers,
                          dropout=dropout_p)
        self.rnn_zero_state = nn.Parameter(
            torch.zeros(n_layers, 1, hidden_size))
        self.output_to_logits = nn.Linear(hidden_size, self.num_classes)
        self.criterion = nn.NLLLoss(reduce=False)
        self.beam_size = beam_size
        self.TRANSCRIPTION_LEN_GUARD = 400

        self.vocabulary = vocabulary

        if lm_file:
            self.lm = fst.Fst.read(lm_file)
            assert self.vocabulary is not None
        else:
            self.lm = None

        self.alphabet_mapping = self.create_alphabet_mapping()

        self.lm_weight = lm_weight
        self.min_attention_pos = min_attention_pos
        self.coverage_tau = coverage_tau
        self.coverage_weight = coverage_weight
        self.length_normalization = length_normalization
        self.keep_eos_score = keep_eos_score
        self.use_graph_search = use_graph_search

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

    def forward(self, encoded, encoded_lens, texts, text_lens,
                return_att_weights=False, return_rnn_states=False, **kwargs):
        '''
        encoded -> max_encoded_len x bs x num_class
        texts -> bs x max_text_len
        '''
        # Init recurrence
        bs, _ = texts.size()

        rnn_state = self.rnn_zero_state.repeat(1, bs, 1)
        att_state, att_weights = self.attn.init_attention(encoded, encoded_lens)

        texts = torch.cat((texts, (torch.zeros(bs, 1).int())), dim=1)
        for bs_it in range(bs):
            texts[bs_it, text_lens[bs_it]] = self.EOS

        max_text_len = texts.size()[1]

        if Globals.cuda:
            texts = texts.cuda()
        texts = texts.long()
        embedded = self.embedding(texts.t())
        # Now embedded are L x B x D

        inputs = embedded.data.new(bs, self.hidden_size).zero_()
        outputs = []
        all_att_weights = []
        all_rnn_states = []
        for targets in embedded:
            att_state, att_weights = self.attn(att_state, rnn_state[0], att_weights)
            # Att_weights are L x B
            all_att_weights.append(att_weights)
            # TODO: Alternative computations, please choose the fastest!
            # context = (att_weights.unsqueeze(2) * encoded).sum(0)
            # context = torch.bmm(att_weights.t().unsqueeze(1),
            #                     encoded.t()).squeeze()
            # context = torch.bmm(encoded.permute(1,2,0),
            #                     att_weights.t().unsqueeze(2)).squeeze(2)
            context = (att_weights.unsqueeze(2) * encoded).sum(0)
            rnn_input = torch.cat((inputs, context), 1).unsqueeze(0)
            output, rnn_state = self.rnn(rnn_input, rnn_state)
            outputs.append(output)
            all_rnn_states += [rnn_state.detach()]
            inputs = targets

        try:
            if logger.is_currently_logging():
                att_weights = torch.cat([aw[:, :1] for aw in all_att_weights],
                                        1).data.cpu().numpy()
                att_weights -= att_weights.min()
                att_weights /= max(1.0, att_weights.max())
                att_weights = np.tile(att_weights[:, :, None], (1, 1, 3))
                logger.log_image('attention_weigths', att_weights)
        except Exception:
            pass

        outputs = torch.cat(outputs)
        logits = self.output_to_logits(outputs)
        logits = logits.permute(1, 0, 2).contiguous()
        loss = F.cross_entropy(logits.view(bs * max_text_len, -1),
                               texts.view(bs * max_text_len),
                               ignore_index=0)

        ret = {'loss': loss}
        if return_att_weights:
            ret['attweights'] = all_att_weights
        if return_rnn_states:
            ret['rnnstates'] = all_rnn_states
        return ret

    def decode(self, encoded, encoded_lens, texts=None,
               text_lens=None, print_debug=False, **kwargs):
        """
        encoded      -> max_len x bs x encoded_size
        encoded_lens -> bs
        texts        -> bs x max_text_len
        text_len     -> bs
        """
        loss = Variable(torch.zeros(1))

        if Globals.cuda:
            loss = loss.cuda()

        bs = encoded.size()[1]

        max_encoded_len = encoded.size()[0]

        batch_size = encoded.size()[1]
        beam_size = self.beam_size

        if self.lm:
            kes = self.keep_eos_score
            if self.use_graph_search:
                beam_search = GraphSearch(
                    self.hash_dec,
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

        rnn_state = self.rnn_zero_state.repeat(1, bs, 1)
        att_state = self.attn.init_attention(encoded, encoded_lens)

        inputs = Variable(torch.zeros(bs * self.beam_size, self.hidden_size))

        if Globals.cuda:
            inputs = inputs.cuda()

        # BS COMBINED WITH BEAM_SIZE
        rnn_state = rnn_state.unsqueeze(2).repeat(1, 1, self.beam_size, 1) \
            .view(1, bs * self.beam_size, -1)
        encoded = encoded.unsqueeze(2).repeat(1, 1, self.beam_size, 1) \
            .view(max_encoded_len, bs * self.beam_size, -1)
        extended_encoded_lens = encoded_lens.clone().unsqueeze(1) \
            .repeat(1, self.beam_size).view(-1)
        att_state, att_weights = self.attn.init_attention(encoded, extended_encoded_lens)

        for iteration_counter in range(self.TRANSCRIPTION_LEN_GUARD):
            att_state, att_weights = self.attn(att_state, rnn_state[0], att_weights)

            context = (att_weights.unsqueeze(2) * encoded).sum(0)
            rnn_input = torch.cat((inputs, context), 1).unsqueeze(0)
            output, rnn_state = self.rnn(rnn_input, rnn_state)
            logits = self.output_to_logits(output)

            new_input, state_mapping = beam_search.step(
                logits, att_weights=att_weights)

            inputs = self.embedding(new_input)
            rnn_state = rnn_state[:, state_mapping]
            att_weights = att_weights[:, state_mapping]

            if beam_search.has_finished():
                print('Finished in {} steps'.format(iteration_counter))
                break

        return {'decoded': beam_search.best_finished,
                'decoded_scores': beam_search.best_finished_scores_elements,
                'loss': torch.Tensor(beam_search.best_finished_scores).mean()}

    def single_step(self,
                    word_input,
                    last_hidden,
                    encoder_outputs,
                    encoded_lens,
                    precomputed_V_enc_out):
        pass
