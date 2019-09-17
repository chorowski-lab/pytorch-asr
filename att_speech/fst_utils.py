from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import gzip
import itertools
import numpy as np
import time
import torch
from torch.autograd import Function
import torch.nn.functional as F

from egs.vocab import Vocab

import pywrapfst as fst
import collections


#
# Functions to aggregate nodes in an FST during decoding
#
def score_nodes(LG, nodes, final=False, use_log_probs=False, expand_symbol=None):
    """Compute the FST score for all paths that end in nodes.

    Args:
        final: if True, try expanding the utterance by one space, then compute
            the sum over paths that end in a final state only
        use_max: if True reduce weights using max (take the most probable one),
            if False reduce using log-sum-exp
        expand_symbol: symbol with which to optionally expand the hypothesis
            (e.g. final space).
    """
    if final:
        if expand_symbol is not None:
            after_space = expand(LG, nodes, LG.input_symbols().find(expand_symbol))
            assert not set(after_space).intersection(set(nodes))
            after_space.update(nodes)
            nodes = after_space
        ws = [float(LG.final(k)) + v for k, v in nodes.items()]
    else:
        ws = nodes.values()
    return reduce_weights(ws, use_log_probs)


def reduce_weights(ws, use_log_probs):
    """Reduce a list of weights.

    Args:
        use_log_probs: if True reduce weights using using -log(exp(-x1) + exp(-x2))
                       if False reduce using min (take the most probable/lowest cost path)
    """
    if len(ws) == 0:
        return float('inf')
    if use_log_probs:
        return -np.logaddexp.reduce(-np.asarray(ws))
    else:
        return min(ws)


# A slower, all FST variant
# def expand_epsilon_alt(LG, nodes, use_log_probs):
#     temp_fst = fst.Fst(arc_type='standard')
#     node_remap = collections.defaultdict(lambda: temp_fst.add_state())
#     reachable = set()
#
#     temp_fst.add_state()
#     temp_fst.set_start(0)
#     for n, w in nodes.items():
#         reachable.add(n)
#         new_st = node_remap[n]
#         temp_fst.add_arc(0, fst.Arc(0, 0, w, new_st))
#         for a in LG.arcs(n):
#             if a.ilabel == 0:
#                 reachable.add(a.nextstate)
#                 temp_fst.add_arc(
#                     new_st,
#                     fst.Arc(0, 0, a.weight, node_remap[a.nextstate]))
#             else:
#                 break
#     if use_log_probs:
#         temp_fst = fst.arcmap(temp_fst, map_type='to_log')
#     distances = fst.shortestdistance(temp_fst)
#     ret = {r: float(distances[node_remap[r]]) for r in reachable}
#     return ret


def expand_epsilon(LG, nodes, use_log_probs):
    """Expand the set of nodes to all nodes reachable via epsilon-transitions.

    Args:
        use_log_probs: how to redce weights, see `reduce_weights`

    Algorithm:
        topo-sort the reachable nodes, then in thos order summ all paths leading to a node.
    """
    P = {}
    done = set()
    Order = []

    def visit(s):
        for a in LG.arcs(s):
            if a.ilabel != 0:
                break  # LG is ilabel sorted, we are past eps-transitions
            n = a.nextstate

            # Add the connection from s to n
            P.setdefault(n, []).append((s, float(a.weight)))

            if n in done:
                continue
            if len(P[n]) > 1:
                # when we start processing n, it has just one connection, if we reach it for a second time
                # this means its a cycle
                raise ValueError("LG has epsilon-cycles!")

            visit(n)
        done.add(s)
        Order.append(s)

    # This loop adds another level of DFS - conceptually imagine that
    # we add a new start node, and eps-arcs from it to all the nodes in our bag
    # and start DFS from this new node.
    for s in nodes:
        P.setdefault(s, []).append((-1, nodes[s]))

        if s in done:
            continue
        if len(P[s]) > 1:
            raise ValueError("LG has epsilon-cycles reachable from start states!")

        visit(s)

    new_nodes = {}
    new_nodes[-1] = 0
    for s in Order[::-1]:
        weights = [new_nodes[p] + w for p, w in P[s]]
        new_nodes[s] = reduce_weights(weights, use_log_probs)
    del new_nodes[-1]
    return new_nodes


def expand_non_epsilon(LG, nodes, label, use_log_probs=False):
    new_nodes = {}
    for n, w in nodes.items():
        for a in LG.arcs(n):
            if a.ilabel == label:
                ns = a.nextstate
                new_w = w + float(a.weight)
                if ns not in new_nodes:
                    new_nodes[ns] = new_w
                else:
                    if use_log_probs:
                        new_nodes[ns] = -np.logaddexp(-new_nodes[ns],
                                                      -new_w)
                    else:
                        new_nodes[ns] = min(new_nodes[ns], new_w)
    return new_nodes


def expand(LG, nodes, label, use_log_probs=False):
    # # Warning, we had a bug here, to repeat use
    # nodes = expand_non_epsilon(LG, nodes, label)
    nodes = expand_non_epsilon(LG, nodes, label, use_log_probs)
    nodes = expand_epsilon(LG, nodes, use_log_probs)
    return nodes

def expand_all(LG, num_classes, nodes, use_log_probs=False):
    global torig
    global talt
    all_new_nodes = [{} for _ in xrange(num_classes)]
    # expand non-epsilons
    for n, w in nodes.items():
        for a in LG.arcs(n):
            new_nodes = all_new_nodes[a.ilabel]
            ns = a.nextstate
            new_w = w + float(a.weight)
            if ns not in new_nodes:
                new_nodes[ns] = new_w
            else:
                if use_log_probs:  # and False # to repeat the bug we had
                    new_nodes[ns] = -np.logaddexp(-new_nodes[ns],
                                                  -new_w)
                else:
                    new_nodes[ns] = min(new_nodes[ns], new_w)
    for i in range(num_classes):
        all_new_nodes[i] = expand_epsilon(LG, all_new_nodes[i], use_log_probs)
    return all_new_nodes

#
# Functions to build FST graphs
#


def build_chain_fst(labels, arc_type='log', vocab=None):
    """
    Build an acceptor for string given by elements of labels.

    Args:
        labels - a sequence of labels in the range 1..S
        arc_type - fst arc type (standard or log)
    Returns:
        FST consuming symbols in the range 1..S.

    Notes:
        Elements of labels are assumed to be greater than zero
        (which maps to blank)!
    """
    C = fst.Fst(arc_type=arc_type)
    weight_one = fst.Weight.One(C.weight_type())
    s = C.add_state()
    C.set_start(s)
    for l in labels:
        s_next = C.add_state()
        C.add_arc(s, fst.Arc(l, l, weight_one, s_next))
        s = s_next
    C.set_final(s)
    C.arcsort('ilabel')
    return C


def fst_to_matrices(g, out_edges=True, nc_weight=None, device=None):
    """
    Encode FST transitions as adjacency lists, prepared as padded matrices.

    Args:
        g - an FST. It shall not have epsilons on input side,
            the outputs are ignored. The ilabels are treated as
            (1 + input symbol)
        out_edges - if True, for each node enumerate outgoing arcs,
                    if False, enumerate incoming arcs
        nc_weight - weight ot assign to masked edges (no connections).
            Should be about -1e20 for single floating precision
        device - device on which to place the tensors

    Returns
        Let K be the maximum out-degree (in-degree when not out_edges), and
        N number of nodes in the FST. 4 tensors are returned:
        states_mat: N x K matrix of successor (predecessor) state id
        ilabels_mat: N x K, labels consumed on the edges
        weights_mat: N x K, log-prob of the edge. Equals to nc_weight if
            the edge is introduced for padding
        terminal_mat: N x 1, log-probs of terminal noeds
            (and neg_inf if node is not terminal)
    """
    if g.start() != 0:
        raise ValueError("FST starting state is not 0, but %d" %
                         (g.start(),))

    is_det = (g.properties(fst.I_DETERMINISTIC, fst.I_DETERMINISTIC) > 0)
    if not is_det:
        raise ValueError("FST is not deterministic")

    if nc_weight is None:
        nc_weight = -fst.Weight.Zero(g.weight_type())
    nc_weight = float(nc_weight)

    edges = [[] for _ in g.states()]
    n = g.num_states()
    terminal_mat = torch.full((n, 1), nc_weight, dtype=torch.float32)
    for prevstate in g.states():
        assert prevstate < n
        term_weight = -float(g.final(prevstate))
        if np.isfinite(term_weight):
            terminal_mat[prevstate, 0] = term_weight
        else:
            terminal_mat[prevstate, 0] = nc_weight

        for a in g.arcs(prevstate):
            ilabel = a.ilabel - 1
            weight = -float(a.weight)
            if ilabel < 0:
                raise ValueError(
                    "FST has eps-transitions (state=%d)" % (prevstate,))

            if out_edges:
                edges[prevstate].append((a.nextstate, ilabel, weight))
            else:
                edges[a.nextstate].append((prevstate, ilabel, weight))
    k = max(len(e) for e in edges)
    states_mat = torch.full((n, k), 0, dtype=torch.int64)
    ilabels_mat = torch.full((n, k), 0, dtype=torch.int64)
    weights_mat = torch.full((n, k), nc_weight, dtype=torch.float32)
    for s1, arcs in enumerate(edges):
        for i, (s2, ilabel, weight) in enumerate(sorted(arcs)):
            states_mat[s1, i] = s2
            ilabels_mat[s1, i] = ilabel
            weights_mat[s1, i] = weight
    if device is not None:
        states_mat = states_mat.to(device)
        ilabels_mat = ilabels_mat.to(device)
        weights_mat = weights_mat.to(device)
        terminal_mat = terminal_mat.to(device)
    return states_mat, ilabels_mat, weights_mat, terminal_mat


try:
    logsumexp = torch.logsumexp
except:
    def logsumexp(inputs, dim=None, keepdim=False, out=None):
        """Numerically stable logsumexp.

        Args:
            inputs: A Variable with any shape.
            dim: An integer.
            keepdim: A boolean.

        Returns:
            Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
        """
        if dim is None:
            inputs, dim = inputs.view(-1), 0
        s, _ = torch.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        if out is not None:
            out[...] = outputs
        return outputs


def path_reduction(
        log_probs, act_lens, graph_matrices, red_kind='logsumexp',
        neg_inf=-1e20):
    """
    Compute a sum of all paths through a graph.

    Args:
        log_probs: T x bs x L tensor of log_probs of emitting symbols
        act_lens: bs tensor of lengths of utternaces
        red_kind: logsumexp / viterbi - chooses between aggregating al paths by
            summing their probabilities (logsumexp of logprobs), or
            by taking the maximally probable one. Also encoded which reduction
            engige ot use:
                logsumexp_fwb forces a forward-backward algo, while
                logsumexp_autodiff uses backward pass using autodiff.
        graphs_matrices: a tuple of four matrices of shape bs x N [x K]
            that encode the transitions and weights in the graph
        neg_inf: what value to use for improbable events (-1e10 or -1e20 are OK)

    Returns:
        tensor of shape bs: a sum of weigths on the maximally probable path
        or on all paths
    """
    if (red_kind == 'logsumexp_fwb' or
            (red_kind == 'logsumexp' and len(graph_matrices) == 8)):
        return path_logsumexp(log_probs, act_lens, graph_matrices, -1e20)

    _, bs, _ = log_probs.size()
    assert graph_matrices[0].size(0) in [1, bs]
    assert all(sm.size(0) == graph_matrices[0].size(0)
               for sm in graph_matrices)
    # This can happen if we get the matrices for full forward-backward
    # and here we only need the ones for worward
    if len(graph_matrices) == 8:
        graph_matrices = graph_matrices[:4]
    if graph_matrices[0].size(0) == 1:
        graph_matrices = [gm.expand(bs, -1, -1) for gm in graph_matrices]
    states_mat, ilabels_mat, weights_mat, terminal_mat = graph_matrices

    _, n, k = states_mat.size()

    if red_kind in ['logsumexp', 'logsumexp_autodiff']:
        # reduction = torch.logsumexp
        reduction = logsumexp
    else:
        assert red_kind in ['viterbi', 'viterbi_autodiff']

        def reduction(t, dim):
            return torch.max(t, dim)[0]

    # a helper to select the next indices for a transition
    def get_idx(m, i):
        _bs = m.size(0)
        return torch.gather(m, 1, i.view(_bs, n * k)).view((_bs, n, k))

    lalpha = torch.full((bs, n), neg_inf, device=log_probs.device)
    lalpha[:, 0] = 0

    # The utterances are sorted according to length descending.
    # Rather than masking, stop updates to alphas when an utterance ends.
    assert act_lens.tolist() == sorted(act_lens, reverse=True)
    last_iter_end = 0
    for bitem in range(bs, 0, -1):
        iter_end = act_lens[bitem - 1]
        for t in range(last_iter_end, iter_end):
            token_probs = (
                get_idx(lalpha[:bitem], states_mat[:bitem]) +
                weights_mat[:bitem] +
                get_idx(log_probs[t, :bitem], ilabels_mat[:bitem]))
            la = reduction(token_probs, dim=-1)
            lalpha = lalpha.clone()
            lalpha[:bitem] = la
        last_iter_end = iter_end

    path_sum = reduction(lalpha + terminal_mat.squeeze(2), dim=-1)
    return path_sum


class PathLogSumExp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, log_probs, act_lens, graph_matrices, neg_inf=-np.inf):
        log_probs = log_probs.detach()
        T, bs, _ = log_probs.size()
        assert graph_matrices[0].size(0) in [1, bs]
        assert all(sm.size(0) == graph_matrices[0].size(0)
                   for sm in graph_matrices)
        if graph_matrices[0].size(0) == 1:
            graph_matrices = [gm.expand(bs, -1, -1) for gm in graph_matrices]
        (states_mat, ilabels_mat, weights_mat, terminal_mat,
         states_mat_out, ilabels_mat_out, weights_mat_out, _
         ) = graph_matrices

        terminal_mat = terminal_mat.squeeze(-1)

        _, n, _ = states_mat.size()

        # a helper to select the next indices for a transition
        def get_idx(m, i):
            _bs = m.size(0)
            return torch.gather(m, 1, i.view(_bs, -1)).view(i.size())

        lalpha = torch.full((bs, n), neg_inf, device=log_probs.device)
        lalpha[:, 0] = 0
        lalpha0 = lalpha.clone()

        lalphas = torch.full((T, bs, n), neg_inf, device=log_probs.device)

        # The utterances are sorted according to length descending.
        # Rather than masking, stop updates to alphas when an utterance ends.
        assert act_lens.tolist() == sorted(act_lens, reverse=True)
        last_iter_end = 0
        for bitem in range(bs, 0, -1):
            iter_end = act_lens[bitem - 1]
            for t in range(last_iter_end, iter_end):
                lalphas[t] = lalpha
                token_probs = weights_mat[:bitem].clone()
                token_probs += get_idx(lalpha[:bitem], states_mat[:bitem])
                token_probs += get_idx(log_probs[t, :bitem],
                                       ilabels_mat[:bitem])
                logsumexp(token_probs, dim=-1, out=lalpha[:bitem])
            last_iter_end = iter_end

        log_cost = logsumexp(lalpha + terminal_mat, dim=-1)

        lbeta = terminal_mat.clone()
        logprobs_grad = torch.zeros_like(log_probs)

        last_iter_end = T
        for bitem in range(1, bs + 1):
            if bitem < bs:
                iter_end = act_lens[bitem]
            else:
                iter_end = 0
            for t in range(last_iter_end - 1, iter_end - 1, -1):
                token_probs = weights_mat_out[:bitem].clone()
                token_probs += get_idx(lbeta[:bitem], states_mat_out[:bitem])
                token_probs += get_idx(log_probs[t, :bitem],
                                       ilabels_mat_out[:bitem])
                logsumexp(token_probs, dim=-1, out=lbeta[:bitem])

                token_probs += (lalphas[t, :bitem] -
                                log_cost[:bitem].unsqueeze(-1)
                                ).unsqueeze(-1)
                token_probs.exp_()

                logprobs_grad[t, :bitem].scatter_add_(
                    1, ilabels_mat_out[:bitem].view(bitem, -1),
                    token_probs.view(bitem, -1))
            last_iter_end = iter_end

            ctx.grads = logprobs_grad

        # approximate the numerical error
        log_cost0 = logsumexp(lalpha0 + lbeta, dim=1)
        if torch.abs(log_cost - log_cost0).max().item() > 1e-3:
            print('forward_backward num error: fwd losses %s bwd losses %s' %
                  (log_cost, log_cost0))
        return log_cost

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output[None, :, None] * ctx.grads,
                None, None, None, None, None, None)


path_logsumexp = PathLogSumExp.apply


def batch_training_graph_matrices(matrices,
                                  nc_weight=-1e20, device='cpu'):
    """
    Combine training matrices for a batch of labels.

    Args:
        matrices: list of tuples of matrices for FSTs
        nc_weight - forarded to fst_to_matrices
        device: pytorch device
    Returns:
        The matrices (see fst_to_matrices) for training FSTs (CTC graphs
        that accept only the trainig utterance), concatenated to a large
        padded matrix of size B x N x K, where B is the batch size,
        N maximum number of states, K maximum degree.
    """
    bs = len(matrices)
    max_n = max([m[0].size(0) for m in matrices])
    max_ks = [max([m[i].size(1) for m in matrices])
              for i in range(len(matrices[0]))]
    batched_matrices = []
    for i, m in enumerate(matrices[0]):
        batched_matrices.append(torch.full(
            (bs, max_n, max_ks[i]),
            0 if m.dtype == torch.int64 else nc_weight,
            dtype=m.dtype, device=device
            ))

    for b, ms in enumerate(matrices):
        for i, m in enumerate(ms):
            batched_matrices[i][b, :m.size(0), :m.size(1)] = m
    return batched_matrices


def make_full_ngram_table(context_order, num_symbols, num_classes):
    if num_symbols is None:
        num_symbols = int(num_classes**(1.0/context_order))
    num_classes = num_symbols ** context_order
    ngram_to_class = []
    for i in range(num_classes):
        ngram_to_class.append(
            i // num_symbols**np.arange(context_order)[::-1] % num_symbols
        )
    ngram_to_class = torch.LongTensor(ngram_to_class)
    return num_symbols, num_classes, ngram_to_class


def setattr_matched(obj, var, val):
    val_orig = getattr(obj, var)
    if val_orig is not None:
        assert val_orig == val, ("The value of %s is %s but should be %s" %
                                 (var, val_orig, val))
    else:
        setattr(obj, var, val)


class BaseGraphGen(object):

    def __init__(self, num_symbols=None, num_classes=None,
                 ngram_to_class_file=None, context_order=None,
                 nc_weight=-1e20, for_forward_only=False,
                 grammar_fst=None, vocabulary=None,
                 **kwargs):
        super(BaseGraphGen, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_symbols = num_symbols
        self.context_order = context_order

        if ngram_to_class_file is None:
            if self.context_order is None:
                self.context_order = 1
            (num_symbols, num_classes, ngram_to_class
             ) = make_full_ngram_table(
                 context_order, num_symbols, num_classes)
            setattr_matched(self, 'num_symbols', num_symbols)
            setattr_matched(self, 'num_classes', num_classes)
            self.ngram_to_class = ngram_to_class
            self.ngrams = ngram_to_class.tolist()
        else:
            with open(ngram_to_class_file, 'r') as f:
                self.ngrams = [[int(n) for n in line.split()] for line in f]
            if self.context_order is None:
                self.context_order = len(self.ngrams[0])
            assert all(len(ngr) == self.context_order for ngr in self.ngrams)
            num_classes = len(self.ngrams)
            setattr_matched(self, 'num_classes', num_classes)
            # NOTE Do not check num_symbols (there might be less in the ngram file)
            self.ngram_to_class = torch.LongTensor(self.ngrams)

        if grammar_fst is not None:
            self.grammar_fst_path = grammar_fst
            self.vocabulary = Vocab.from_file(vocabulary)
        else:
            self.grammar_fst_path = None

        self.nc_weight = nc_weight
        self.decoding_fst = self.get_decoding_fst()
        # self.num_symbols = num_symbols
        # self.num_classes = num_classes
        self.decoding_mats = {}
        self.for_forward_only = for_forward_only

    def get_transcript_fst(self, labels):
        if type(labels) is torch.Tensor:
            labels = labels.cpu().numpy()
        if np.any(labels > self.num_symbols):
            labels = labels % self.num_symbols
            if getattr(self, '_print_bigram_data_warn', True):
                print("Got input symbol larger than num_symbols. "
                      "Are you using a Bigram dataset with FSTs?")
                self._print_bigram_data_warn = False
        return build_chain_fst(labels)

    def get_training_fst(self, labels):
        return fst.compose(self.decoding_fst,
                           self.get_transcript_fst(labels))

    def get_training_matrices_batch(self, labels, label_lens,
                                    device='cpu'):
        matrices = [
            self.get_training_matrices(labels[i, :label_lens[i]], 'cpu')
            for i in range(len(labels))]
        return batch_training_graph_matrices(
            matrices, self.nc_weight, device=device)

    def get_grammar_fst(self):
        print("load grammar fst from:", self.grammar_fst_path)
        with gzip.open(self.grammar_fst_path) as gf:
            g_fst = fst.Fst.read_from_string(gf.read())
        g_fst = fst.arcmap(g_fst, map_type="to_log")
        # remap symbols to network vocab
        n_syms = fst.SymbolTable()
        for i, s in enumerate(self.vocabulary.itos):
            s = {' ': '<spc>',
                 '<pad>': '<eps>'
                 }.get(s, s)
            assert i == n_syms.add_symbol(s)
        # This stops a warninrg and is harmless - these will not occur anyway
        n_syms.add_symbol('<s>')
        n_syms.add_symbol('</s>')
        g_fst.relabel_tables(new_isymbols=n_syms, new_osymbols=n_syms)
        return g_fst

    def get_decoding_fst(self):
        hc_fst = self.get_hc_fst()
        if self.grammar_fst_path:
            ret = fst.compose(hc_fst, self.get_grammar_fst())
            ret = fst.rmepsilon(fst.determinize(ret))
            return ret
        else:
            return hc_fst

    def get_hc_fst(self):
        """Build the H * C FST, i.e. the HMM and Context transducer.
        """
        raise NotImplementedError()

    def get_training_matrices(self, labels, device='cpu'):
        # if labels.shape[0] == 2:
        #     # XXX XXX XXX
        #     print('WARNING: transcription too short to build 3-grams', labels)
        #     labels = np.ones((3,), dtype=labels.dtype) * labels[0]
        train_fst = self.get_training_fst(labels)
        matrices = fst_to_matrices(
            train_fst, out_edges=False,
            nc_weight=self.nc_weight, device=device)
        if not self.for_forward_only:
            matrices += fst_to_matrices(
                train_fst, out_edges=True,
                nc_weight=self.nc_weight, device=device)
        return matrices

    def get_decoding_matrices(self, device='cpu', out_edges=False):
        key = str(device)
        ret = self.decoding_mats.get(key)
        if ret is not None:
            return ret
        self.decoding_mats[key] = [
            m.unsqueeze(0) for m in fst_to_matrices(
                self.decoding_fst, out_edges=False,
                nc_weight=self.nc_weight, device=device)]
        if not self.for_forward_only:
            self.decoding_mats[key] += [
                m.unsqueeze(0) for m in fst_to_matrices(
                    self.decoding_fst, out_edges=True,
                    nc_weight=self.nc_weight, device=device)]
        return self.decoding_mats[key]


def build_ctc_mono_decoding_fst(S, arc_type='log', add_syms=False):
    """
    Build a monophone CTC decoding fst.
    Args:
        S - number of monophones
        arc_type - log or standard. Gives the interpretation of the FST.
    Returns:
        an FST that accepts all sequences over [1,..,S]^* and returns
        shorter ones with duplicates and blanks removed.

        The input labels are shifted by one, so that there are no epsilon
        transitions.
        The output labels are not (blank is zero), allowing one to read out
        the label sequence easily.
    """
    CTC = fst.Fst(arc_type=arc_type)
    weight_one = fst.Weight.One(CTC.weight_type())

    for s in range(S):
        s1 = CTC.add_state()
        assert s == s1
        CTC.set_final(s1)
    CTC.set_start(0)

    for s in range(S):
        # transitions out of symbol s
        # self-loop, don't emit
        CTC.add_arc(s, fst.Arc(s + 1, 0, weight_one, s))
        for s_next in range(S):
            if s_next == s:
                continue
            # transition to next symbol
            CTC.add_arc(s, fst.Arc(s_next + 1, s_next, weight_one, s_next))
    CTC.arcsort('olabel')

    if add_syms:
        in_syms = fst.SymbolTable()
        in_syms.add_symbol('<eps>', 0)
        in_syms.add_symbol('B', 1)
        for s in range(1, S):
            in_syms.add_symbol(chr(ord('a') + s - 1), s + 1)
        out_syms = fst.SymbolTable()
        out_syms.add_symbol('<eps>', 0)
        for s in range(1, S):
            out_syms.add_symbol(chr(ord('a') + s - 1), s)
        CTC.set_input_symbols(in_syms)
        CTC.set_output_symbols(out_syms)
    return CTC


def build_ctc_bigram_decoding_fst(S, arc_type='log',
                                  allow_nonblank_selfloops=True,
                                  use_contextual_blanks=False,
                                  loop_using_symbol_repetitions=False,
                                  eval_repeats_in_context=False,
                                  add_syms=False):
    """
    Build a CTC decoding FST that reads in biphones and produces decodings.

    Args:
        S - number of monophones
        arc_type - log or standard. Gives the interpretation of the FST.
        allow_nonblank_selfloops - if we allow self loop arcs consuming
           symbols other than blanks
        use_contextual_blanks - is there a single blank (symbol 0), or
            several blanks - one in each context
        loop_using_symbol_repetitions - do we allow self loops, by emitting
            a symbol from its own context
    Returns:
        an FST that accepts all sequences over [1,..,S^2]^* and returns
        shorter ones with duplicates and blanks removed.

        The input labels are (context*S + label) + 1, there are no epsilon
        transitions.
        The output labels are not (blank is zero), allowing one to read out
        the label sequence easily.

    Note: for proper operation (i.e. to make sure that probability of all
      paths sums to 1), set loop_using_symbol_repetitions = True!!!!!!!!!
    """
    assert not (eval_repeats_in_context and loop_using_symbol_repetitions)
    assert not eval_repeats_in_context
    if loop_using_symbol_repetitions:
        allow_nonblank_selfloops = False
        use_contextual_blanks = True

    CTC = fst.Fst(arc_type=arc_type)
    weight_one = fst.Weight.One(CTC.weight_type())

    for s in range(S ** 2):
        s1 = CTC.add_state()
        assert s == s1
        CTC.set_final(s1)
    # blank after blank is the final state
    CTC.set_start(0)

    def get_input_sym(c, let):
        if let != 0 or use_contextual_blanks:
            # read label in context, shifted by 1 to account for eps
            return c * S + let + 1
        else:
            # read the global blank
            return 0 + 1

    for s1 in range(S ** 2):
        c1 = s1 // S  # prev context
        l1 = s1 % S  # prev symbol

        # emit the xy non-blank self loop of y in the context of x
        # please note that:
        # - xB and xx self loops will be handled later
        self_loop = None
        if allow_nonblank_selfloops and l1 != 0 and c1 != l1:
            # repeat the last symbol in its context
            in_symbol = get_input_sym(c1, l1)
            CTC.add_arc(s1, fst.Arc(in_symbol, 0, weight_one, s1))
            self_loop = s1

        # determine the next context
        if l1 == 0:  # last symbol read was a blank, no change in context
            c2 = c1
        else:
            c2 = l1

        for l2 in range(S):
            s2 = c2 * S + l2
            # assert that we didn't emit the self loop before
            assert not self_loop == s2

            if (l2 == 0 or s1 == s2 or
                    (loop_using_symbol_repetitions and l1 == l2)):
                out_s = 0
            else:
                out_s = l2

            CTC.add_arc(
                s1,
                fst.Arc(get_input_sym(c2, l2), out_s, weight_one, s2))
    CTC.arcsort('olabel')

    if add_syms:
        in_syms = fst.SymbolTable()
        in_syms.add_symbol('<eps>', 0)
        for s1 in range(S):
            for s2 in range(S):
                in_syms.add_symbol(
                    ('B' if s1 ==0 else chr(ord('a') + s1 - 1)) +
                    ('B' if s2 ==0 else chr(ord('a') + s2 - 1)),
                    s1 * S + s2 + 1)
        out_syms = fst.SymbolTable()
        out_syms.add_symbol('<eps>', 0)
        for s in range(1, S):
            out_syms.add_symbol(chr(ord('a') + s - 1), s)
        CTC.set_input_symbols(in_syms)
        CTC.set_output_symbols(out_syms)

    return CTC


def reachable_trigrams(s1, trigrams):
    blank = 0
    out_eps = 0

    def fits(s, pattern):
        pattern = [blank if c == 'B' else 1 for c in pattern]
        return all([s[i] == blank == pattern[i] or s[i] != blank != pattern[i]
                    for i in range(3)])

    # Accumulate pairs (next_state, olabel)
    # Every state allows to loop
    reachable = [(s1, out_eps)]

    if fits(s1, 'xyz'):
        # yBz  : eps
        reachable.append(((s1[1], blank, s1[2]), out_eps))
        assert reachable[-1][0] in trigrams
        # yzw  : z
        # yzB  : z
        reachable += [(s2, s1[2]) for s2 in trigrams
                      if s1 != s2 and s2[:2] == s1[1:] and s1 != s2]
    elif fits(s1, 'BBx'):
        # Bxy  : x
        reachable += [(s2, s1[2]) for s2 in trigrams
                      if s1 != s2 and s2[:2] == s1[1:] and s2[2] != blank]
    elif fits(s1, 'Bxy'):
        # xyz  : y
        reachable += [(s2, s1[2]) for s2 in trigrams
                      if s1 != s2 and s2[:2] == s1[1:] and blank not in s2]
        # xBy  : eps
        reachable.append(((s1[1], blank, s1[2]), out_eps))
        assert reachable[-1][0] in trigrams
    elif fits(s1, 'yzB'):
        # zBB  : eps
        reachable.append(((s1[1], blank, blank), out_eps))
        assert reachable[-1][0] in trigrams
    elif fits(s1, 'xBy'):
        # xyz  : y
        reachable += [(s2, s1[2]) for s2 in trigrams
                      if s1 != s2 and s1[0] == s2[0] and s1[2] == s2[1] and blank not in s2]
    elif fits(s1, 'BBB'):
        # BBx  : eps
        reachable += [(s2, out_eps) for s2 in trigrams if s1 != s2 and s1[1:] == s2[:2]]
    return list(set(reachable))


def build_ctc_trigram_decoding_fst_alan(S, trigrams, arc_type='log'):
    """
    Args:
    """
    CTC = fst.Fst(arc_type=arc_type)
    weight_one = fst.Weight.One(CTC.weight_type())

    # Translate from tuples of letter indices to indices
    # NOTE: Not all trigrams are present, states are enumerated without gaps.
    # in_ind = lambda s: sum([S**(2-i) * c for i,c in enumerate(s)]) + 1
    in_ind = lambda s: tg2state[s] + 1
    out_ind = lambda s: s # out_ind(s) + 1
    state_ind = lambda s: tg2state[s]

    # Need a hashable type
    trigrams = [tuple(tg) for tg in trigrams]

    # Build
    tg2state = {}
    for i,tg in enumerate(trigrams):
        s1 = CTC.add_state()
        CTC.set_final(s1)
        assert s1 == i
        tg2state[tg] = i

    assert trigrams[0] == (0, 0, 0)  # blank, blank, blank
    CTC.set_start(0)

    for i,s1 in enumerate(trigrams):
        if i % 1000 == 0:
            print(i)
        for s2, out in reachable_trigrams(s1, trigrams):
            CTC.add_arc(state_ind(s1),
                        fst.Arc(in_ind(s2), out_ind(out), weight_one, state_ind(s2)))

    CTC.arcsort('olabel')
    return CTC  #, tg2state


WARNINGS = set()


def build_ctc_trigram_decoding_fst_v2(S, trigrams, arc_type='log',
                                      use_context_blanks=False,
                                      prevent_epsilons=True,
                                      determinize=False,
                                      add_syms=False):
    """
    Args:
    """
    CTC = fst.Fst(arc_type=arc_type)
    weight_one = fst.Weight.One(CTC.weight_type())

    # Need a hashable type
    trigrams = sorted([tuple(tg) for tg in trigrams])

    # Translate from tuples of letter indices to indices
    # NOTE: Not all trigrams are present, states are enumerated without gaps.
    # in_ind = lambda s: sum([S**(2-i) * c for i,c in enumerate(s)]) + 1
    def in_ind(s):
        if not use_context_blanks and s[1] == 0:
            return tg2state[(0, 0, 0,)] + 1
        return tg2state[s] + 1

    def out_ind(s):
        return s

    def state_ind(s):
        return tg2state[s]

    # Build
    tg2state = {}
    # Add a final looping BBB state
    # for i, tg in enumerate(itertools.chain(trigrams, [(0, 0, 0)])):
    for i, tg in enumerate(trigrams):
        s1 = CTC.add_state()
        if tg[-1] == 0:
            CTC.set_final(s1)
        assert s1 == i
        tg2state[tg] = i

    assert trigrams[0] == (0, 0, 0)  # blank, blank, blank
    CTC.set_start(0)

    if 0:
        # Add a special state to handle empty labels
        s_final = CTC.add_state()
        CTC.set_final(s_final)
        CTC.add_arc(0, fst.Arc(in_ind((0, 0, 0)), 0, weight_one, s_final))
        CTC.add_arc(s_final, fst.Arc(in_ind((0, 0, 0)), 0, weight_one, s_final))

#     # Add the self-loop in the extra final state
#     CTC.add_arc(tg2state[(0, 0, 0)], fst.Arc(in_ind(s1), 0, weight_one, tg2state[(0, 0, 0)]))

    for i1, s1 in enumerate(trigrams):
        # Handle the self loop. Please note, that it correctly handles the start and final
        # (0, 0, 0) states
        if s1 != (0, 0, 0):
            CTC.add_arc(i1, fst.Arc(in_ind(s1), 0, weight_one, i1))

        if s1[1] == 0:
            base_low = (s1[0], s1[2], 0)
            base_high = (s1[0], s1[2], np.inf)
        else:
            base_low = (s1[1], s1[2], 0)
            base_high = (s1[1], s1[2], np.inf)

        base_index = bisect.bisect_left(trigrams, base_low)
        # assert trigrams[base_index] == base_low
        if not trigrams[base_index] == base_low:
            global WARNINGS
            if base_low not in WARNINGS:
                print ("missing trigram ", base_low)
                WARNINGS.add(base_low)
        high_index = bisect.bisect_left(trigrams, base_high)
        for s2 in set(itertools.chain(trigrams[base_index:high_index],
                                      [(s1[1], 0, s1[2])])):
            # self loop is already handled
            if s1 == s2:
                continue
            if s2 == (0, 0, 0):
                continue
            if s1 != (0, 0, 0):
                # once we emit the final blank, we need to terminate
                if s1[-1] == 0 and s2[-1] != 0:
                    continue
                if s1[-2] == (0, 0) and s2[-1] != 0:
                    continue
                # we can't emit a starting blank, unless we start
                if s2[0] == 0 and s1[0] != 0:
                    continue
            if s1 != (0, 0, 0) or prevent_epsilons:
                in_label = in_ind(s2)
            else:
                in_label = 0
            out_label = out_ind(s2[1])
            CTC.add_arc(i1, fst.Arc(in_label, out_label,
                                    weight_one, state_ind(s2)))
    print("Dec g is det?",
          CTC.properties(fst.I_DETERMINISTIC, fst.I_DETERMINISTIC) > 1)
    print("Determinizing the decoding graph")
    CTC = CTC.rmepsilon()
    if determinize:
        CTC = fst.determinize(CTC)
    print("Decoding graph nas %d states and max %d out-degree" %
          (CTC.num_states(),
           max([CTC.num_arcs(s) for s in range(CTC.num_states())])))

    CTC.arcsort('olabel')
    if add_syms:
        in_syms = fst.SymbolTable()
        in_syms.add_symbol('<eps>', 0)
        max_sym = 0
        for tg, key in tg2state.items():
            sym = ''.join(['B' if c == 0 else chr(ord('a') + c - 1)
                           for c in tg])
            max_sym = max(max_sym, max(tg))
            in_syms.add_symbol(sym, key + 1)

        out_syms = fst.SymbolTable()
        out_syms.add_symbol('<eps>', 0)
        for s in range(1, max_sym + 1):
            out_syms.add_symbol(chr(ord('a') + s - 1), s)
        CTC.set_input_symbols(in_syms)
        CTC.set_output_symbols(out_syms)

    return CTC


class CTCGraphGen(BaseGraphGen):
    def __init__(self, context_order=None, graph_build_args=None, **kwargs):
        assert context_order in (1, 2, 3)
        self.graph_build_args = graph_build_args or {}
        super(CTCGraphGen, self).__init__(context_order=context_order,
                                          **kwargs)

    def get_hc_fst(self):
        if self.context_order == 1:
            return build_ctc_mono_decoding_fst(self.num_symbols)
        elif self.context_order == 2:
            return build_ctc_bigram_decoding_fst(self.num_symbols,
                                                 **self.graph_build_args)
        elif self.context_order == 3:
            return build_ctc_trigram_decoding_fst_v2(
                self.num_symbols, self.ngrams, **self.graph_build_args)
