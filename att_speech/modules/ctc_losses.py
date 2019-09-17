#
# Jan Chorowski 2018, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import time
import torch
from torch.autograd import Function
import torch.nn.functional as F

import pywrapfst as fst

from att_speech.fst_utils import (
    logsumexp,
    build_chain_fst,
    build_ctc_mono_decoding_fst, build_ctc_bigram_decoding_fst,
    fst_to_matrices, batch_training_graph_matrices,
    path_reduction, path_logsumexp, CTCGraphGen)


def get_normalized_acts(acts, acts_lens,
                        num_symbols, context_order, normalize_by_dim,
                        normalize_logits=True):
    assert context_order == 1 or num_symbols
    del acts_lens  # unused
    if normalize_by_dim:
        act_size = acts.size()
        new_act_size = (act_size[:2] + (num_symbols,) * context_order)
        acts = acts.view(*new_act_size)
        acts = torch.nn.functional.log_softmax(
            acts, normalize_by_dim + 2)
        acts = acts.view(*act_size)
    elif normalize_logits:
        acts = torch.nn.functional.log_softmax(acts, dim=-1)
    return acts


def ctc_loss(acts, labels, act_lens, label_lens,
            num_symbols=None, context_order=1, normalize_by_dim=None,
            allow_nonblank_selfloops=True,
            loop_using_symbol_repetitions=False,
            eval_repeats_in_context=False,
            other_data_in_batch=None):
    for condition in [context_order == 1,
                      normalize_by_dim is None,
                      not eval_repeats_in_context,
                      allow_nonblank_selfloops,
                      not loop_using_symbol_repetitions,
                      not other_data_in_batch]:
        assert condition, "Option not supported in this loss"
    # F.ctc_loss doesn't check these conditions and may segfault later on
    assert acts.size(0) == act_lens[0]
    assert labels.max() < acts.size(2)
    return F.ctc_loss(F.log_softmax(acts, dim=-1).contiguous(), labels.cuda(),
                      act_lens, label_lens, reduction='mean',
                      zero_infinity=False)


def get_CTC_matrices_mono(L, S):
    """
    Generate transition matrices for monophone CTC.
    Args:
        L: sequence of >0 integer token ids (0is reserved for blank)
        S: total number of symbols (with blank)
    """
    N = 2 * L.size(0) + 1
    A = torch.zeros(N, N)
    B = torch.zeros(S, N)
    # The first optional blank
    A[0, 0] = 1.0  # loop on the first blank
    A[0, 1] = 1.0  # go from blank to 1. symbol
    B[0, 0] = 1.0
    for i, l in enumerate(L):
        base_state = 2 * i + 1
        B[l, base_state] = 1      # this state emits l
        B[0, base_state + 1] = 1  # this state + 1 emits a blank
        A[base_state, base_state] = 1  # self loop
        A[base_state, base_state + 1] = 1  # can emit blank
        A[base_state + 1, base_state + 1] = 1  # blank can loop
        if base_state + 2 < N:
            # can go from blank to next label
            A[base_state + 1, base_state + 2] = 1
            if L[i+1] != l:
                # label changed, can directly emit
                A[base_state, base_state + 2] = 1
    return A, B


def get_CTC_decoding_matrices_mono(S):
    """
    Generate matrix of allowed transitions and emissions for monophone CTC.

    Args:
        S - number of symbols

    Please note that this function does not remove blanks nor duplicates.
    """
    A = torch.ones(S, S)  # can go from any symbol to any symbol
    B = torch.eye(S, S)  # each symbol emits self
    return A, B


def get_CTC_matrices_bicontext(L, S, eval_repeats_in_context=False,
                               allow_nonblank_selfloops=True,
                               loop_using_symbol_repetitions=False):
    """
    Generate transition matrices for biphone CTC.

    A context-specific blank is emitted before each symbol.
    The last blank is taken from the last symbol's context.

    Args:
        L: sequence of >0 integer token ids (0is reserved for blank)
        S: total number of symbols (with blank)
    """
    if loop_using_symbol_repetitions:
        assert not eval_repeats_in_context
        return get_CTC_matrices_bicontext_looprepeats(L, S)
    L = L % S  # remove the context - we will add it below ourselves
    N = 2 * L.size(0) + 1
    A = torch.zeros(N, N)
    B = torch.zeros(S**2, N)
    # The first optional blank

    context = 0
    for i, li in enumerate(L):
        prev_blank_state = 2 * i
        token_state = 2 * i + 1

        # emissions
        # the preceding contextualized blank
        B[context * S + 0, prev_blank_state] = 1
        # this symbol's emission
        B[context * S + li, token_state] = 1

        # transitions
        A[prev_blank_state, prev_blank_state] = 1  # blank can loop
        # blank can transition to the token
        A[prev_blank_state, token_state] = 1
        if allow_nonblank_selfloops:
            A[token_state, token_state] = 1  # token can loop
        if token_state + 1 < N:  # there is a following blank
            # token can transition to next blank
            A[token_state, token_state + 1] = 1
        # there is a following token and it is different
        if (token_state + 2 < N):
            if eval_repeats_in_context:
                is_rep = ((context * S + li) == (li * S + L[i+1]))
            else:
                is_rep = (li == L[i+1])
            if not is_rep:
                # token can transition directly to next token
                A[token_state, token_state + 2] = 1
        context = li
    # emit the last blank from last symbol's context
    B[context * S + 0, -1] = 1
    A[-1, -1] = 1  # last blank can loop
    return A, B


def get_CTC_matrices_bicontext_looprepeats(L, S):
    """
    Generate transition matrices for biphone CTC.

    A context-specific blank is emitted before each symbol.
    The last blank is taken from the last symbol's context.

    Args:
        L: sequence of >0 integer token ids (0is reserved for blank)
        S: total number of symbols (with blank)
    """
    L = L % S  # remove the context - we will add it below ourselves
    N = 3 * L.size(0) + 1
    A = torch.zeros(N, N)
    B = torch.zeros(S**2, N)

    context = 0
    # first blank
    B[0*S + 0, 0] = 1.0  # emits the blank

    A[0, 0] = 1.0  # first blank can loop
    A[0, 1] = 1.0  # first blank can transition to first emission

    for i, label in enumerate(L):
        entry_state = 3*i + 1
        looping_state = entry_state + 1
        blank_state = entry_state + 2
        next_token_state = entry_state + 3

        # emissions
        # first emission comes from old context
        B[context * S + label, entry_state] = 1

        context = label
        B[context * S + label, looping_state] = 1  # this symbol's emission
        B[context * S + 0, blank_state] = 1  # this symbol's emission

        if i + 1 < len(L) and label != L[i+1]:
            can_emit_next_token = True
        else:
            can_emit_next_token = False

        # transitions
        A[entry_state, looping_state] = 1.0  # can go to looping state
        A[entry_state, blank_state] = 1.0  # can go to blank
        if can_emit_next_token:
            A[entry_state, next_token_state] = 1.0

        A[looping_state, looping_state] = 1.0
        A[looping_state, blank_state] = 1.0
        if can_emit_next_token:
            A[looping_state, next_token_state] = 1.0

        A[blank_state, blank_state] = 1.0
        if i + 1 < len(L):
            A[blank_state, next_token_state] = 1.0

    return A, B


def get_CTC_decoding_matrices_bicontext(S, eval_repeats_in_context=False,
                                        allow_nonblank_selfloops=True):
    """
    Generate matrix of allowed transitions and emissions for monophone CTC.

    Args:
        S - number of symbols

    Please note that this function does not remove blanks nor duplicates.
    It only ensures that all symbls are emited from the correct context.
    """
    # assert eval_repeats_in_context, "Case for False not implemented"  # XXX
    assert allow_nonblank_selfloops, "Case for False not implemented"
    S2 = S**2
    A = torch.zeros(S, S, S, S)  # in context, in sym, out context, out sym
    B = torch.eye(S2, S2)

    # a blank maintains its context for all subsequent symbols
    for context in range(S):
        A[context, 0, context, :] = 1.0
    # any symbl but the blank sets the context for the next char
    for symbol in range(1, S):
        # all transitions that go from this symbol to another one change
        # the context
        A[:, symbol, symbol, :symbol] = 1.0
        A[:, symbol, symbol, symbol:] = 1.0

        if not eval_repeats_in_context:
            A[:, symbol, :, symbol] = 0.0

        # using this symbol in the past context does not introduce a new one
        if allow_nonblank_selfloops:
            for c in range(S):
                A[c, symbol, c, symbol] = 1.0

    A = A.view(S2, S2)
    return A, B


class RawGenericCTC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, A, B,
                enter_first_num=2, terminal_last_num=2):
        logsum = np.logaddexp.reduce
        T, S = log_probs.size()
        N = A.size(0)
        assert A.size(1) == N
        assert B.size(0) == S
        assert B.size(1) == N
        assert (B.sum(0) == 1).all()
        B = B.detach().numpy()
        log_probs = log_probs.detach().numpy()
        lPB = log_probs.dot(B)

        lA = np.log(A.numpy())

        lalpha = np.empty((T, N))
        lalpha.fill(-np.inf)
        lalpha[0, :enter_first_num] = 0

        for t in range(1, T):
            logsum((lalpha[t-1] + lPB[t-1])[:, None] + lA,
                   axis=0, out=lalpha[t])
            # lalpha[t, :(t-T)*2] = -np.inf

        lbeta = np.empty((T, N))
        lbeta.fill(-np.inf)
        lbeta[-1, -terminal_last_num:] = 0

        for t in range(T - 2, -1, -1):
            logsum(lA + (lbeta[t + 1] + lPB[t + 1])[None, :],
                   axis=1, out=lbeta[t])
            # lbeta[t, (t + 1) * 2:] = -np.inf

        log_costs = -logsum(lPB + lalpha + lbeta, axis=-1)
        log_cost = log_costs.mean()
        num_err = log_costs.max() - log_costs.min()
        if num_err > 0.1:
            print("Numerical instability in Raw CTC of ", num_err)

        ldiffs = -np.exp(lalpha + lbeta + log_cost
                         ).dot(B.T)  # this is the diff wrt. probs
        # print(ldiffs[:2])
        ldiffs *= np.exp(log_probs)  # diff through exponentiation

        # print(ldiffs)

        ctx.grads = torch.from_numpy(ldiffs).float()
        return torch.tensor(log_cost).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.grads, None, None, None, None


raw_generic_ctc = RawGenericCTC.apply


class RawGenericCTCBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, A, B, act_lens, label_lens,
                enter_first_num=2, terminal_last_num=2):
        T, bs, S = log_probs.size()
        assert A.size(0) == B.size(0) == bs
        N = A.size(1)
        assert A.size(2) == N
        assert B.size(1) == S
        assert B.size(2) == N
        assert (B.sum(1) <= 1).all()
        B = B.detach()
        log_probs = log_probs.detach()
        lPB = log_probs.transpose(0, 1).matmul(B).transpose(0, 1).contiguous()
        lA = A.log()

        mask = (torch.arange(T, dtype=torch.int32).unsqueeze(1) < act_lens
                ).to(A.device).unsqueeze(-1)

        lalpha = torch.full((T, bs, N), -np.inf, device=A.device)
        lalpha[0, :, :enter_first_num] = 0

        for t in range(1, T):
            la = logsumexp(
                (lalpha[t-1] + lPB[t-1]).unsqueeze(2) + lA,
                dim=1)
            lalpha[t] = torch.where(mask[t], la, lalpha[t-1])

        lbeta = torch.full((T, bs, N), -np.inf, device=A.device)
        for i in range(bs):
            lbeta[-1, i, (label_lens[i] - 1) * terminal_last_num + 1:] = 0
            if act_lens[i] < lPB.size(0):
                lPB[act_lens[i]:, i, :] = lPB[act_lens[i]-1, i, :]

        for t in range(T - 2, -1, -1):
            lb = logsumexp(
                lA + (lbeta[t + 1] + lPB[t + 1]).unsqueeze(1),
                dim=2)
            lbeta[t] = torch.where(mask[t + 1], lb, lbeta[t + 1])

        log_costs = -logsumexp(lPB + lalpha + lbeta, dim=-1)

        log_cost = log_costs.mean(0)
        num_err = (log_costs.max(0)[0] - log_costs.min(0)[0])
#         if torch.any(num_err > 0.1):
#             print("Numerical instability in Raw CTC of ", num_err)

        # diff wrt probs
        ldiffs = -(lalpha + lbeta + log_cost[None, :, None]
                   ).exp().transpose(0, 1).matmul(
                       B.transpose(1, 2)).transpose(0, 1)
        ldiffs *= log_probs.exp()  # diff through exponentiation
        ldiffs *= mask.float()  # zero diff wrt unused frames

        ctx.grads = ldiffs
        return log_cost

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output[None, :, None] * ctx.grads,
                None, None, None, None, None, None)


raw_generic_ctc_batch = RawGenericCTCBatch.apply


def cpp_raw_genertic_viterbi(log_probs, A, B,
                             start_states=None, terminal_states=None):
    from att_speech.modules.ctc_losses_cpp import cext
    if start_states is None:
        start_states = []
    if terminal_states is None:
        terminal_states = []
    log_probs = log_probs.detach().numpy()
    A = A.detach().numpy()
    B = B.detach().numpy()
    lPB = log_probs.dot(B)
    lA = np.log(A)
    cost, sol = cext.compute(log_probs, A, B,
                             np.array(start_states, dtype=np.float64),
                             np.array(terminal_states, dtype=np.float64),
                             lPB, lA)
    return cost, torch.IntTensor(sol)


def raw_generic_viterbi(log_probs, A, B,
                        start_states=None, terminal_states=None):
    T, S = log_probs.size()
    N = A.size(0)
    assert A.size(1) == N
    assert B.size(0) == S
    assert B.size(1) == N
    assert np.all(B.sum(0) == 1)
    B = B.detach().numpy()
    log_probs = log_probs.detach().numpy()
    lPB = log_probs.dot(B)
    lA = np.log(A.numpy())

    lalpha = np.empty((T, N))
    lalpha.fill(-np.inf)
    lalpha[0] = lPB[0]
    if start_states is not None:
        mask = np.empty((N))
        mask.fill(-np.inf)
        mask[start_states] = 0
        lalpha[0] += mask
    idx = []

    for t in range(1, T):
        at = lalpha[t-1][:, None] + lA
        ati = np.argmax(at, axis=0)
        idx.append(ati)
        lalpha[t] = at[ati, np.arange(N)] + lPB[t]

    if terminal_states is not None:
        mask = np.empty((N))
        mask.fill(-np.inf)
        mask[terminal_states] = 0
        lalpha[-1] += mask

    sol = np.empty(T, dtype='int32')
    st_to_sym = B.argmax(0)
    st = lalpha[-1].argmax()
    cost = -lalpha[-1][st]
    sol[-1] = st_to_sym[st]

    for t in range(T-1, 0, -1):
        st = idx[t-1][st]
        sol[t-1] = st_to_sym[st]

    return cost, torch.from_numpy(sol)


def mono_CTC_loss(log_probs, labels,
                  allow_nonblank_selfloops=True,
                  loop_using_symbol_repetitions=False,
                  eval_repeats_in_context=False):
    assert allow_nonblank_selfloops
    assert not loop_using_symbol_repetitions
    assert not eval_repeats_in_context
    S = log_probs.size(1)
    A, B = get_CTC_matrices_mono(labels, S)
    if log_probs.is_cuda:
        A = A.cuda()
        B = B.cuda()
    return raw_generic_ctc(log_probs, A, B)


def mono_CTC_viterbi_align(log_probs, labels):
    S = log_probs.size(1)
    A, B = get_CTC_matrices_mono(labels, S)
    N = A.size(0)
    return raw_generic_viterbi(log_probs, A, B,
                               start_states=[0, 1],
                               terminal_states=[N-2, N-1])


def mono_CTC_viterbi_decode(log_probs):
    S = log_probs.size(1)
    A, B = get_CTC_decoding_matrices_mono(S)
    N = A.size(0)
    return raw_generic_viterbi(log_probs, A, B,
                               start_states=None, terminal_states=None)


def bi_CTC_loss(log_probs, labels, **kwargs):
    S = int(np.sqrt(log_probs.size(1)))
    A, B = get_CTC_matrices_bicontext(labels, S, **kwargs)
    if log_probs.is_cuda:
        A = A.cuda()
        B = B.cuda()
    if kwargs.get('loop_using_symbol_repetitions'):
        num_terminal = 3
    else:
        num_terminal = 2
    return raw_generic_ctc(log_probs, A, B, 2, num_terminal)


def bi_CTC_viterbi_align(log_probs, labels, **kwargs):
    S = int(np.sqrt(log_probs.size(1)))
    A, B = get_CTC_matrices_bicontext(labels, S, **kwargs)
    N = A.size(0)
    return raw_generic_viterbi(log_probs, A, B,
                               start_states=[0, 1],
                               terminal_states=[N-2, N-1])


def bi_CTC_viterbi_decode(log_probs, **kwargs):
    S = int(np.sqrt(log_probs.size(1)))
    A, B = get_CTC_decoding_matrices_bicontext(S, **kwargs)
    return raw_generic_viterbi(log_probs, A, B,
                               start_states=range(S), terminal_states=None)


def ctc_raw_loss(acts, labels, act_lens, label_lens,
                 num_symbols=0, context_order=1, normalize_by_dim=None,
                 allow_nonblank_selfloops=True,
                 loop_using_symbol_repetitions=False,
                 eval_repeats_in_context=False,
                 other_data_in_batch=None,
                 **kwargs):
    assert not other_data_in_batch
    log_probs = get_normalized_acts(acts, act_lens, num_symbols,
                                    context_order, normalize_by_dim,
                                    normalize_logits=True)
#     debug_softmax = True
#     if debug_softmax:
#         log_probs = torch.nn.functional.log_softmax(log_probs)
    log_probs = log_probs.cpu()
    # probs.cpu()
    if context_order == 1:
        lfun = mono_CTC_loss
    else:
        assert context_order == 2
        assert normalize_by_dim == 1
        assert num_symbols > 0
        lfun = bi_CTC_loss
    ctc_losses = []
    lab_ends = F.pad(label_lens, (1, 0)).cumsum(0)
    for i in range(act_lens.size(0)):
        ctc_losses.append(
            lfun(log_probs[:act_lens[i], i, :],
                 labels[lab_ends[i]:lab_ends[i + 1]],
                 allow_nonblank_selfloops=allow_nonblank_selfloops,
                 loop_using_symbol_repetitions=loop_using_symbol_repetitions,
                 eval_repeats_in_context=eval_repeats_in_context,
                 **kwargs)
            )
#         if debug_softmax:
#             if context_order == 2:
#                 ctc_losses[-1] -= (np.log(
#                     num_symbols) * (context_order - 1) * act_lens[i]).float()
    ctc_losses = torch.stack(ctc_losses)
    return ctc_losses


def ctc_raw_loss_batch(acts, labels, act_lens, label_lens,
                       num_symbols=0, context_order=1, normalize_by_dim=None,
                       allow_nonblank_selfloops=True,
                       loop_using_symbol_repetitions=False,
                       eval_repeats_in_context=False,
                       other_data_in_batch=None,
                       **kwargs):
    assert not other_data_in_batch
    log_probs = get_normalized_acts(acts, act_lens, num_symbols,
                                    context_order, normalize_by_dim,
                                    normalize_logits=True)
    assert not (eval_repeats_in_context and loop_using_symbol_repetitions)
    # log_probs = log_probs.cpu()
    if context_order == 1:
        assert not loop_using_symbol_repetitions
        mat_fun = get_CTC_matrices_mono
        S = log_probs.size(2)
    else:
        assert context_order == 2
        assert normalize_by_dim == 1
        assert num_symbols > 0
        kwargs['allow_nonblank_selfloops'] = allow_nonblank_selfloops
        kwargs['loop_using_symbol_repetitions'] = loop_using_symbol_repetitions
        kwargs['eval_repeats_in_context'] = eval_repeats_in_context
        mat_fun = get_CTC_matrices_bicontext
        S = int(np.sqrt(log_probs.size(2)))

    if kwargs.get('loop_using_symbol_repetitions'):
        num_terminal = 3
    else:
        num_terminal = 2

    bs = log_probs.size(1)
    N = label_lens.max().item() * num_terminal + 1
    A = torch.zeros(bs, N, N)
    B = torch.zeros(bs, log_probs.size(2), N)

    lab_ends = F.pad(label_lens, (1, 0)).cumsum(0)
    for i in range(bs):
        _A, _B = mat_fun(labels[lab_ends[i]:lab_ends[i + 1]], S, **kwargs)
        A[i, :_A.size(0), :_A.size(1)] = _A
        B[i, :_B.size(0), :_B.size(1)] = _B
    B = B.to(log_probs.device)
    A = A.to(log_probs.device)
    ctc_losses = raw_generic_ctc_batch(
        log_probs, A, B, act_lens, label_lens, 2, num_terminal)
    return ctc_losses


#
# CTC loss compatible with reductions of FSTs
#


def ctc_fst_loss(acts, labels, act_lens, label_lens,
                 num_symbols=0, context_order=1, normalize_by_dim=None,
                 allow_nonblank_selfloops=True,
                 loop_using_symbol_repetitions=False,
                 other_data_in_batch=None,
                 eval_repeats_in_context=False,
                 neg_inf=-1e20,
                 ):
    log_probs = get_normalized_acts(acts, act_lens, num_symbols,
                                    context_order, normalize_by_dim,
                                    normalize_logits=True)
    # log_probs = acts
    assert not (eval_repeats_in_context and loop_using_symbol_repetitions)
    S = log_probs.size(2)

    if other_data_in_batch and 'graph_matrices' in other_data_in_batch:
        graph_matrices = [gm.to(log_probs.device) for gm in
                          other_data_in_batch['graph_matrices']]
    else:
        gg_kwargs = {}
        if context_order == 2:
            S = int(np.sqrt(S))
            gg_kwargs = dict(
                allow_nonblank_selfloops=allow_nonblank_selfloops,
                loop_using_symbol_repetitions=loop_using_symbol_repetitions,
                eval_repeats_in_context=eval_repeats_in_context)
        graph_gen = CTCGraphGen(
            S, context_order, for_forward_only=False,
            graph_build_args=gg_kwargs)
        labels_b = torch.zeros((label_lens.size(0), label_lens.max()),
                               dtype=labels.dtype)
        ls = 0
        for b in range(label_lens.size(0)):
            le = ls + label_lens[b]
            labels_b[b, :label_lens[b]] = labels[ls:le]
            ls = le
        graph_matrices = graph_gen.get_training_matrices_batch(
            labels_b, label_lens, device=log_probs.device)

    return -path_reduction(log_probs, act_lens, graph_matrices,
                           neg_inf=neg_inf)


__ctc_fst_cache = {}


def ctc_fst_global_loss(acts, labels, act_lens, label_lens,
                        num_symbols=0, context_order=1, normalize_by_dim=None,
                        allow_nonblank_selfloops=True,
                        loop_using_symbol_repetitions=False,
                        other_data_in_batch=None,
                        eval_repeats_in_context=False,
                        neg_inf=-1e20,
                        ):
    tg = time.time()
    log_probs = acts
    S = log_probs.size(2)
    global __ctc_fst_cache
    key = (context_order, allow_nonblank_selfloops,
           loop_using_symbol_repetitions, eval_repeats_in_context)
    if key not in __ctc_fst_cache:
        if context_order == 1:
            decoding_g = build_ctc_mono_decoding_fst(S)
        else:
            assert context_order == 2
            S = int(np.sqrt(S))
            decoding_g = build_ctc_bigram_decoding_fst(
                S,
                allow_nonblank_selfloops=allow_nonblank_selfloops,
                loop_using_symbol_repetitions=loop_using_symbol_repetitions,
                eval_repeats_in_context=eval_repeats_in_context)

        denominator_matrices = [
            m.unsqueeze(0) for m in
            fst_to_matrices(decoding_g, out_edges=False,
                            nc_weight=neg_inf, device=log_probs.device)]

        __ctc_fst_cache[key] = (decoding_g, denominator_matrices)
    decoding_g, denominator_matrices = __ctc_fst_cache[key]

    if other_data_in_batch and 'graph_matrices' in other_data_in_batch:
        numerator_matrices = [gm.to(log_probs.device) for gm in
                              other_data_in_batch['graph_matrices']]
    else:
        numerator_matrices = build_ctc_training_graph_matrices(
            S, labels, label_lens, decoding_g, out_edges=False,
            nc_weight=neg_inf, device=log_probs.device)

    ts = time.time()
    numerator_loss = -path_reduction(
        log_probs, act_lens, numerator_matrices,
        red_kind='logsumexp', neg_inf=neg_inf)
    tn = time.time()
    denominator_loss = -path_reduction(
        log_probs, act_lens, denominator_matrices,
        red_kind='logsumexp', neg_inf=neg_inf)
    print("global loss: [time: gprep: %f num %f, den  %f;"
          "loss: num %g, den %g]" % (
              ts - tg, tn - ts, time.time() - tn,
              numerator_loss.sum().item(), denominator_loss.sum().item()))
    return numerator_loss - denominator_loss
