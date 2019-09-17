import torch
import numpy as np
import time
from att_speech import fst_utils


itos = ['<pad>', '<unk>', '<spc>', 'E', 'T', 'A', 'O', 'N', 'I', 'S', 'R', 'H',
    'L', 'D', 'C', 'U', 'M', 'P', 'F', 'G', 'Y', 'B', 'W', 'V', 'K', '.', "'",
    'X', 'Q', '~', 'J', ',', '-', 'Z', '"', '*', ':', '(', ')', '?', '!', '&',
    ';', '/', '{', '}', '<', '>', '_']


class BeamSearch(object):
    '''
     local_scores -> 1 x [self.batch_size x self.beam_size] x num_classes
     scores -> [self.batch_size x self.beam_size]
     estimations -> [self.batch_size x self.beam_size] ...
     att_weights -> T x [self.batch_size x self.beam_size]
     best finished -> list of decoded vectors
     best_finished_scores -> list of scalars
    '''
    def __init__(self, batch_size, beam_size, device,
                 num_classes, length_normalization,
                 keep_eos_score=False):

        self.scores = torch.zeros(batch_size * beam_size, device=device)
        self.estimations = None
        self.finished_count = [0 for i in range(batch_size)]
        self.best_finished = [[] for i in range(batch_size)]
        self.best_finished_scores = [float('-inf')] * batch_size
        self.best_finished_scores_elements = {
            'acoustic': self.best_finished_scores
        }
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.length_normalization = length_normalization
        self.min_eos = None
        self.keep_eos_score = keep_eos_score
        self.coverage = None
        self.attentions = None

        self.print_debug = False
        self.gather_attentions = False

    def to_text(self, est):
        return ''.join(itos[e] if itos[e] != '<spc>' else ' ' for e in est)

    def _local_prunning(self, local_scores, global_score):
        return global_score
        # Local pruning: don't allow low local score choices
        # best_in_beam = torch.max(local_scores, 1)[0].unsqueeze(1)
        # bt = self.branching_threshold
        # return torch.where(
        #     local_scores + bt >= best_in_beam,
        #     global_score, torch.ones_like(global_score) * float('-inf'))

    def _save_best_finished(self, global_scores):
        '''
        global_scores - [beam_size x batch_size, num_classes]

        For each batch:
            take extended-beam with best EOS score, and check if it is new best
        '''
        scores = global_scores[:, -1].contiguous().view(self.batch_size, -1)
        EOSscores = (
            scores /
            (self.estimations.size(1) ** self.length_normalization))
        is_eos_best = (
            torch.argmax(global_scores, dim=1) == global_scores.size(1) - 1
        )
        for batch_id in range(self.batch_size):
            ind = torch.argmax(EOSscores[batch_id])
            if is_eos_best[batch_id] and self.finished_count[batch_id] <= self.beam_size:
                self.finished_count[batch_id] += 1
                if self.best_finished_scores[batch_id] < EOSscores[batch_id][ind]:
                    self.best_finished_scores[batch_id] = EOSscores[batch_id][ind]
                    self.best_finished_scores_elements['acoustic'][batch_id] = \
                        scores[batch_id][ind].item()
                    self.best_finished[batch_id] = \
                        self.estimations[batch_id * self.beam_size + ind]

    def _get_topk(self, scores):
        if self.beam_size < scores.size(1):
            new_scores, best_it = torch.topk(scores, self.beam_size, dim=1)
        else:
            new_scores, best_it = torch.topk(scores, scores.size(1), dim=1)

            to_repeat = self.beam_size - scores.size(1)

            no_scores = torch.ones_like(new_scores[:, -1:]) * float('-inf')
            no_scores = no_scores.repeat(1, to_repeat)

            new_scores = torch.cat((new_scores, no_scores), dim=1)
            best_it = torch.cat((
                best_it,
                best_it[:, -1:].repeat(1, to_repeat)), dim=1)
        return new_scores, best_it

    def _add_new_column_to_estimations(self):
        new_col = torch.zeros((self.batch_size * self.beam_size, 1),
                              device=self.scores.device).long()
        if self.estimations is None:
            self.estimations = new_col
        else:
            self.estimations = torch.cat((self.estimations, new_col), dim=1)

    def _compute_new_beam(self, best_it):
        best_beams = best_it // (self.num_classes - 1)
        best_letters = best_it % (self.num_classes - 1)

        new_beam_mapping = torch.zeros(self.batch_size * self.beam_size).long()
        new_estimations = self.estimations.clone()

        for batch_id in range(self.batch_size):
            for beam_id in range(self.beam_size):
                current_id = batch_id * self.beam_size + beam_id
                new_beam_id = best_beams[batch_id, beam_id]
                new_id = batch_id * self.beam_size + new_beam_id
                new_beam_mapping[current_id] = new_id
                new_estimations[current_id, :] = self.estimations[new_id]
                new_estimations[current_id, -1] = \
                    best_letters[batch_id, beam_id]
        return new_beam_mapping, new_estimations, best_letters

    def _do_ignore_eos(self, global_scores):
        global_scores = global_scores[:, :-1]
        global_scores = (
            global_scores.contiguous().view(self.batch_size, -1))

        if self.estimations is None:
            global_scores = global_scores[:, :self.num_classes - 1]
        return global_scores

    def _get_eos_score_from_previous_frame(self, unnormalized_local_scores):
        if self.min_eos is not None:
            unnormalized_local_scores[:, -1] = torch.where(
                unnormalized_local_scores[:, -1] > self.min_eos,
                unnormalized_local_scores[:, -1], self.min_eos
            )
        self.min_eos = unnormalized_local_scores[:, -1]
        return unnormalized_local_scores

    def _update_eos_scores_with_new_beam(self, beam_mapping):
        self.min_eos = self.min_eos[beam_mapping]

    def step(self, logits, *args, **kwargs):
        local_scores = logits.squeeze(0)

        local_scores = torch.nn.functional.log_softmax(local_scores, dim=1)
        global_scores = (local_scores +
                         self.scores.unsqueeze(1).repeat(1, self.num_classes))

        global_scores = self._local_prunning(local_scores, global_scores)

        if self.estimations is not None:
            self._save_best_finished(global_scores)

        # ignore EOS from now on
        global_scores = self._do_ignore_eos(global_scores)

        # new_scores, best_it =
        #   torch.topk(global_scores, self.beam_size, dim=1)
        new_scores, best_it = self._get_topk(global_scores)

        new_scores = new_scores.view(-1)

        self._add_new_column_to_estimations()

        (new_beam_mapping, new_estimations,
         best_letters) = self._compute_new_beam(best_it)

        self.scores = new_scores
        self.estimations = new_estimations
        return best_letters.view(-1), new_beam_mapping

    def has_finished(self):
        return all(self.finished_count[i] >= self.beam_size for i in range(self.batch_size))
        # return (self.scores == float('-inf')).all()

    def get_graph(self):
        return None


class BeamSearchLM(BeamSearch):
    def __init__(self, lm, lm_weight, alphabet_mapping, min_attention_pos,
                 coverage_tau, coverage_weight, *args, **kwargs):
        super(BeamSearchLM, self).__init__(*args, **kwargs)
        self.lm = lm
        self.fst_states = [
            {self.lm.start(): 0} for _ in xrange(self.beam_size)]
        self.alphabet_mapping = alphabet_mapping
        self.lm_weight = lm_weight
        self.finished = []
        self.min_attention_pos = min_attention_pos
        self.coverage_tau = coverage_tau
        self.coverage_weight = coverage_weight
        self.best_finished_scores_elements = {
            'acoustic': [0],
            'lm': [0],
        }
        if self.coverage_weight > 0:
            self.best_finished_scores_elements['coverage'] = [0]
        assert self.batch_size == 1

    def _step_lm(self):
        lm_scores = torch.zeros((self.beam_size, self.num_classes))
        if self.lm_weight == 0:
            return lm_scores, self.fst_states
        all_fst_states = []
        for beam_id, beam_nodes in enumerate(self.fst_states):
            current_beam_fst_states = []
            new_fst_states_shuffled = fst_utils.expand_all(
                self.lm, self.num_classes, beam_nodes, True)
            for letter in range(self.num_classes):
                current_beam_fst_states.append(
                    new_fst_states_shuffled[self.alphabet_mapping[letter]])
                nxt = fst_utils.reduce_weights(
                    current_beam_fst_states[-1].values(), True)
                nxt = min(1e20, nxt)
                lm_scores[beam_id, letter] = -self.lm_weight * nxt
            all_fst_states.append(current_beam_fst_states)
        return lm_scores, all_fst_states

    def _add_finished(self, total_scores, score_elements, att_weights):
        '''
        total_scores - [beam_size, num_classes]

        For each batch:
            take extended-beam with best EOS score, and check if it is new best
        '''
        # EOSscores = global_scores[:, -1]
        min_pos = self.min_attention_pos*att_weights.size(0)
        EOSscores = (
            total_scores[:, -1] /
            (self.estimations.size(1) ** self.length_normalization))
        # EOSscores = scores, when we take EOS
        looking_far_enough = (
            att_weights.argmax(dim=0) > min_pos)
        is_eos_best = (
            torch.argmax(total_scores, dim=1) == total_scores.size(1) - 1
        )
        added = False
        finish_mask = [False] * EOSscores.size(0)
        for beam in range(EOSscores.size(0)):
            if (is_eos_best[beam] and looking_far_enough[beam]
                    and EOSscores[beam].item() > -1e10):
                finish_mask[beam] = True
                self.finished += [(EOSscores[beam], self.estimations[beam], beam)]
                added = True
                if self.print_debug:
                    sent = self.to_text(self.estimations[beam])  #''.join(itos[e] if itos[e] != '<spc>' else ' ' for e in self.estimations[beam])
                    print('Added to finshed {} {}'.format(sent, EOSscores[beam]))
        if self.finished and added:
            self.finished = sorted(
                self.finished, key=lambda x: x[0].item(), reverse=True)
            self.finished = self.finished[:self.beam_size]
            if self.finished[0][0] > self.best_finished_scores[0]:
                self.best_finished_scores[0] = self.finished[0][0]
                self.best_finished[0] = self.finished[0][1]
                self.best_finished_scores_elements = {
                    k: [v[self.finished[0][2], -1].item()] for k, v in score_elements.iteritems()
                }
        return finish_mask

    def step(self, logits, att_weights, print_lm=None):
        if self.coverage_weight > 0:
            if self.coverage is None:
                self.coverage = att_weights.clone()
            else:
                self.coverage += att_weights
        if self.gather_attentions:
            if self.attentions is None:
                self.attentions = att_weights.clone().unsqueeze(-1)
            else:
                self.attentions = torch.cat(
                    (self.attentions, att_weights.unsqueeze(-1)), dim=-1)

        local_scores = logits.squeeze(0)
        if self.keep_eos_score:
            local_scores = self._get_eos_score_from_previous_frame(
                local_scores)
        local_scores = torch.nn.functional.log_softmax(local_scores, dim=1)
        acoustic_scores = (local_scores +
                           self.scores.unsqueeze(1).repeat(1, self.num_classes))

        lm_scores, all_fst_states = self._step_lm()
        lm_scores = lm_scores.to(acoustic_scores.device)

        score_elements = {
            'acoustic': acoustic_scores.clone(),
            'lm': lm_scores.clone(),
        }

        total_scores = acoustic_scores + lm_scores

        coverage_scores = 0
        if self.coverage_weight > 0:
            coverages = (
                (self.coverage > self.coverage_tau).sum(dim=0).float())
            # print(coverage_scores)
            coverage_scores = self.coverage_weight * (
                coverages.unsqueeze(1).repeat(1, self.num_classes))
            total_scores += coverage_scores
            score_elements['coverage'] = coverage_scores

        if self.estimations is not None:
            self._add_finished(
                total_scores,
                score_elements,
                att_weights)

        # ignore EOS from now on
        total_scores = self._do_ignore_eos(total_scores)
        sel_tot_scores, best_it = self._get_topk(total_scores)

        # Truncate EOS, select best based on topk
        acoustic_scores = acoustic_scores[
            :, :-1].contiguous().view(self.batch_size, -1)

        new_scores = acoustic_scores[:, best_it[0]]
        if self.beam_size >= acoustic_scores.size(1):
            new_scores[:, -(self.beam_size - acoustic_scores.size(1)):] = (
                float('-inf'))
        self.scores = new_scores.view(-1)

        new_fst_states = []
        if self.lm_weight != 0:
            for ind in best_it[0]:
                ind_beam = ind // (self.num_classes - 1)
                ind_letter = ind % (self.num_classes - 1)
                new_fst_states.append(all_fst_states[ind_beam][ind_letter])
        self.fst_states = new_fst_states

        self._add_new_column_to_estimations()
        (new_beam_mapping, new_estimations,
         best_letters) = self._compute_new_beam(best_it)
        self.estimations = new_estimations

        if self.keep_eos_score:
            self._update_eos_scores_with_new_beam(new_beam_mapping)

        if self.coverage_weight > 0:
            self.coverage = self.coverage[:, new_beam_mapping]

        if self.attentions is not None:
            self.attentions = self.attentions[:, new_beam_mapping, :]

        if self.print_debug:
            print "%s a:%.3f l:%.f c:%.3f (%d)" % (
                self.to_text(self.estimations[0]),
                self.scores[0],
                -fst_utils.reduce_weights(self.fst_states[0].values(), True) if self.lm_weight > 0 else 0,
                (self.coverage > self.coverage_tau).sum(0)[0].item(),
                len(self.finished))
        return best_letters.view(-1), new_beam_mapping

    def debug_estimations(self):
        for est in self.estimations:
            print(self.to_text(est))

    def has_finished(self):
        return len(self.finished) >= self.beam_size


class RescoreSearchLM(BeamSearchLM):
    def __init__(self, sentence, *args, **kwargs):
        super(RescoreSearchLM, self).__init__(*args, **kwargs)
        self.sentence = sentence
        self.gather_attentions = True
        assert self.beam_size == 1

    def _get_topk(self, scores):
        let_id = self.estimations.size(1) if self.estimations is not None else 0
        if let_id < len(self.sentence):
            cur_id = self.sentence[let_id]
        else:
            cur_id = 0
        return (scores[:, cur_id:(cur_id+1)],
                torch.LongTensor([[cur_id]]).to(scores.device))

    def _add_finished(self, global_scores, score_elements, att_weights):
        if self.estimations.size(1) == len(self.sentence):
            min_pos = self.min_attention_pos*att_weights.size(0)
            EOSscores = (
                global_scores[:, -1] /
                (self.estimations.size(1) ** self.length_normalization))
            # EOSscores = scores, when we take EOS
            looking_far_enough = (
                att_weights.argmax(dim=0) > min_pos)
            is_eos_best = (
                torch.argmax(global_scores, dim=1) == global_scores.size(1) - 1
            )
            # print('Would allow EOS? {}'.format(looking_far_enough.item() and is_eos_best.item()))

            self.finished += [(EOSscores[0], self.estimations[0], 0)]

            if self.finished[0][0] > self.best_finished_scores[0]:
                self.best_finished_scores[0] = self.finished[0][0]
                self.best_finished[0] = self.finished[0][1]
                self.best_finished_scores_elements = {
                    k: [v[self.finished[0][2], -1].item()] for k, v in score_elements.iteritems()
                }


class GraphSearch(BeamSearchLM):
    def __init__(self, hash_dec, merge_threshold, *args, **kwargs):
        super(GraphSearch, self).__init__(*args, **kwargs)
        self.graph = [{} for _ in xrange(self.batch_size)]
        self.hash_dec = hash_dec
        self.merge_threshold = merge_threshold

    def att_prod(self, x, y):
        # print (torch.stack([x,y], dim = 1))
        return torch.sum(torch.min(x, y))

    def is_prefix(self, l1, l2):
        if len(l1) > len(l2):
            return False
        l2 = l2[:len(l1)]
        return all(l1 == l2)

    def step(self, logits, att_weights, print_lm=None):
        if self.coverage_weight > 0:
            if self.coverage is None:
                self.coverage = att_weights.clone()
            else:
                self.coverage += att_weights

        local_scores = logits.squeeze(0)
        if self.keep_eos_score:
            local_scores = self._get_eos_score_from_previous_frame(
                local_scores)
        local_scores = torch.nn.functional.log_softmax(local_scores, dim=1)
        acoustic_scores = (local_scores +
                           self.scores.unsqueeze(1).repeat(1, self.num_classes))

        lm_scores, all_fst_states = self._step_lm()
        lm_scores = lm_scores.to(acoustic_scores.device)

        score_elements = {
            'acoustic': acoustic_scores.clone(),
            'lm': lm_scores.clone(),
        }

        total_scores = acoustic_scores + lm_scores

        coverage_scores = 0
        if self.coverage_weight > 0:
            coverages = (
                (self.coverage > self.coverage_tau).sum(dim=0).float())
            # print(coverage_scores)
            coverage_scores = self.coverage_weight * (
                coverages.unsqueeze(1).repeat(1, self.num_classes))
            total_scores += coverage_scores
            score_elements['coverage'] = coverage_scores

        if self.estimations is not None:
            finish_mask = self._add_finished(
                total_scores,
                score_elements,
                att_weights)
        else:
            finish_mask = [False] * self.beam_size

        if self.beam_size > 1 and self.estimations is not None:
            for batch_id in range(self.batch_size):
                for beam_id in range(self.beam_size):
                    if not finish_mask[beam_id]:
                        continue
                    current_id = batch_id * self.beam_size + beam_id
                    hist_hash = self.hash_dec(self.estimations[current_id])
                    li = self.graph[batch_id].get(hist_hash, [])
                    for id, (score, atts, (fst, fin, coverage), ests, uplink) in enumerate(li):
                        if (ests.shape == self.estimations[beam_id].shape) and (ests == self.estimations[beam_id]).all():
                            li[id] = (score, atts, (fst, True, coverage), ests, uplink)

        # ignore EOS from now on
        total_scores = self._do_ignore_eos(total_scores)
        sel_tot_scores, best_it = self._get_topk(total_scores)

        # Truncate EOS, select best based on topk
        acoustic_scores = acoustic_scores[
            :, :-1].contiguous().view(self.batch_size, -1)

        new_scores = acoustic_scores[:, best_it[0]]
        if self.beam_size >= acoustic_scores.size(1):
            new_scores[:, -(self.beam_size - acoustic_scores.size(1)):] = (
                float('-inf'))
        #self.scores = new_scores.view(-1)
        new_scores = new_scores.view(-1)

        new_tot_scores = total_scores[:, best_it[0]]
        if self.beam_size >= total_scores.size(1):
            new_tot_scores[:, -(self.beam_size - total_scores.size(1)):] = (
                float('-inf'))
        new_tot_scores = new_tot_scores.view(-1)

        new_fst_states = []
        if self.lm_weight != 0:
            for ind in best_it[0]:
                ind_beam = ind // (self.num_classes - 1)
                ind_letter = ind % (self.num_classes - 1)
                new_fst_states.append(all_fst_states[ind_beam][ind_letter])
        self.fst_states = new_fst_states

        self._add_new_column_to_estimations()
        (new_beam_mapping, new_estimations,
         best_letters) = self._compute_new_beam(best_it)
        self.estimations = new_estimations

        if self.keep_eos_score:
            self._update_eos_scores_with_new_beam(new_beam_mapping)

        if self.coverage_weight > 0:
            self.coverage = self.coverage[:, new_beam_mapping]

        if self.beam_size > 1:
            for batch_id in range(self.batch_size):
                # if any(finish_mask):
                #     print(finish_mask)
                #     for beam_id in range(self.beam_size):
                #         current_id = batch_id * self.beam_size + beam_id
                #         print(self.to_text(new_estimations[current_id]))
                #     print('----')
                for beam_id in range(self.beam_size):
                    current_id = batch_id * self.beam_size + beam_id
                    if new_scores[current_id] == float('-inf'):
                        continue
                    hist_hash = self.hash_dec(new_estimations[current_id])
                    li = self.graph[batch_id].get(hist_hash, [])
                    new_uplink = None
                    for id, (score, atts, (fst, fin, coverage), ests, uplink) in enumerate(li):
                        if uplink is not None:
                            # dead branch
                            continue
                        if new_fst_states and set(new_fst_states[current_id].keys()) != fst:
                            # different fst state, ignore
                            continue
                        prod = self.att_prod(atts,
                                             att_weights[:, current_id])
                        if prod < self.merge_threshold:
                            # it's a different branch, ignore it
                            continue
                        # merge branches
                        if score / len(ests) ** self.length_normalization \
                                >= (new_tot_scores[current_id] /
                                    new_estimations.size(1)
                                    ** self.length_normalization):
                            # old branch is better, uplink
                            # and delete the new one
                            # print('Old')
                            # print('Merging {} with {}'.format(
                            #     self.to_text(new_estimations[current_id]),
                            #     self.to_text(ests)
                            # ))
                            new_scores[current_id] = float('-inf')
                            new_tot_scores[current_id] = float('-inf')
                            new_uplink = id
                            break
                        else:
                            # print('Merging {} with {}'.format(
                            #     self.to_text(ests),
                            #     self.to_text(new_estimations[current_id])
                            # ))
                            # print('New')
                            li[id] = (score, atts, (fst, fin, coverage), ests, len(li))
                            # delete all the descandants of the old branch:
                            # iterate over all
                            # active branches and remove all
                            # that have ests as prefix
                            for oth_beam_id in range(self.beam_size):
                                if oth_beam_id == beam_id:
                                    continue
                                oth_id = (batch_id * self.beam_size
                                          + oth_beam_id)
                                if self.is_prefix(ests,
                                                  new_estimations[oth_id]):
                                    new_scores[oth_id] = float('-inf')
                                    new_tot_scores[oth_id] = float('-inf')

                    li.append((new_tot_scores[current_id],
                               att_weights[:, current_id],
                               (
                                set(new_fst_states[current_id].keys()) if new_fst_states else set(),
                                False, #finish_mask[current_id],
                                None, #new_coverage[current_id]
                               ),
                               new_estimations[current_id], new_uplink))
                    self.graph[batch_id][hist_hash] = li

        self.scores = new_scores

        if self.print_debug:
            print "%s a:%.3f l:%.f c:%.3f (%d)" % (
                self.to_text(self.estimations[0]),
                self.scores[0],
                -fst_utils.reduce_weights(self.fst_states[0].values(), True),
                (self.coverage > self.coverage_tau).sum(0)[0],
                len(self.finished))
        return best_letters.view(-1), new_beam_mapping

    def get_graph(self):
        # move uplinks to the sinks
        for hmap in self.graph:
            for h, l in hmap.items(): # hist-hash, l
                # print('HMAP {}'.format(h))
                for i in range(len(l)): # over all nodes in given hist-hash
                    if l[i][-1] is not None: # uplink is not none (uplink == None --> no merge)
                        t = i
                        while l[t][-1] is not None:
                            t = l[t][-1]
                        # print('{} -> {}'.format(i, t))
                        l[i] = l[i][:-1] + (t,) # uplink to highest node
                for i in range(len(l)):
                    tup_encoded = tuple(l[i][3].tolist()) # estimations
                    # hashed decoded, letter
                    l[i] = l[i] + (hash(tup_encoded), l[i][3][-1])

        # Structure of the graph: G = (V, E), where V is a set of vertices,
        # each being a tuple (hash, label (letter),
        # score (cumulated)) and E is set of edges, each being a tuple
        # (hash1, hash2, type).
        G = []
        for hmap in self.graph:
            V = [(hash(()), '<sos>', 0.0, 0., False)]
            valid = set()
            valid.add(hash(()))
            E = []
            # add vertices
            for h, l in hmap.items():
                for sc, atts, (fsts, fin, cov), ests, uplink, ests_hash, label in l:
                    if uplink is None:
                        valid.add(ests_hash)
                        V.append((ests_hash, label.item(), sc.item(), cov, fin))
            # add edges
            for h, l in hmap.items():
                for sc, atts, (fsts, fin, cov), ests, uplink, ests_hash, label in l:
                    parent = hash(tuple(ests[:-1].tolist()))
                    me = ests_hash
                    edg_type = 'normal'
                    if uplink is not None:
                        me = l[uplink][5]
                        edg_type = 'merged'
                    if parent in valid and me in valid:
                        E.append((parent, me, edg_type))
            G.append({'V': V, 'E': E})
        return G
