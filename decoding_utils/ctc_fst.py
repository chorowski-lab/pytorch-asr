#!/usr/bin/env python
#
# Jan Chorowski 2018, UWr
#
'''

'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse

import pywrapfst as fst

from att_speech.fst_utils import build_ctc_trigram_decoding_fst_v2


def read_net_vocab(net_vocab_file):
    with open(net_vocab_file) as f:
        net_vocab = [s.strip()
                     for s in f.readlines()
                     # skip empty line
                     if s.strip()]
    return net_vocab


def gen_unigram_graph(net_vocab_file, token_file, out_file,
                      add_final_space=False,
                      allow_nonblank_selfloops=True,
                      use_contextual_blanks=None,
                      loop_using_symbol_repetitions=None):
    del use_contextual_blanks  # unused, does not apply to this model
    del loop_using_symbol_repetitions  # unused, does not apply to this model
    net_vocab = read_net_vocab(net_vocab_file)
    print("net vocab", net_vocab)

    CTC = fst.Fst(arc_type='standard')
    CTC_os = fst.SymbolTable.read_text(token_file)
    CTC_is = fst.SymbolTable()
    CTC_is.add_symbol('<eps>', 0)
    for i, s in enumerate(net_vocab):
        CTC_is.add_symbol(s, i + 1)

    CTC.set_input_symbols(CTC_is)
    CTC.set_output_symbols(CTC_os)

    after_blank = CTC.add_state()
    CTC.set_start(after_blank)
    CTC.set_final(after_blank)

    l2s = {'<pad>': after_blank}

    for i in range(CTC_is.num_symbols()):
        i = CTC_is.get_nth_key(i)
        let = CTC_is.find(i)
        if l in ('<pad>', '<eps>'):
            continue
        l2s[let] = CTC.add_state()
        CTC.set_final(l2s[let])

    weight_one = fst.Weight.One('tropical')

    final_space_arc = None
    if add_final_space:
        final_space = CTC.add_state()
        CTC.set_final(final_space)
        final_space_arc = fst.Arc(
            CTC_is.find('<eps>'), CTC_os.find('<spc>'),
            weight_one, final_space)

    os_eps = CTC_os.find('<eps>')

    for let, s in l2s.items():
        in_label = CTC_is.find(let)
        out_label = os_eps if let == '<pad>' else CTC_os.find(let)

        # Self-loop, don't emit
        if let == '<pad>' or allow_nonblank_selfloops:
            CTC.add_arc(s, fst.Arc(in_label, os_eps, weight_one, s))

        # Transition from another state - this emits
        for l2, s2 in l2s.items():
            if let == l2:
                continue
            CTC.add_arc(s2, fst.Arc(in_label, out_label, weight_one, s))

        # Optional transition to emit the final space
        if final_space_arc is not None:
            CTC.add_arc(s, final_space_arc)

    CTC.arcsort('olabel')
    CTC.write(out_file)


def gen_bigram_graph(net_vocab_file, token_file, out_file,
                     add_final_space=False,
                     allow_nonblank_selfloops=True,
                     use_contextual_blanks=False,
                     loop_using_symbol_repetitions=False,
                     ):
    if loop_using_symbol_repetitions:
        allow_nonblank_selfloops = False
        use_contextual_blanks = True

    net_vocab = read_net_vocab(net_vocab_file)
    print("net vocab", net_vocab)

    CTC = fst.Fst(arc_type='standard')
    CTC_os = fst.SymbolTable.read_text(token_file)
    CTC.set_output_symbols(CTC_os)
    os_eps = CTC_os.find('<eps>')

    N = len(net_vocab)

    for i in range(N ** 2):
        s = CTC.add_state()
        CTC.set_final(s)
        assert i == s

    # blank after blank is the start state
    CTC.set_start(0)

    weight_one = fst.Weight.One('tropical')

    final_space_arc = None
    if add_final_space:
        final_space = CTC.add_state()
        CTC.set_final(final_space)
        final_space_arc = fst.Arc(
            0, CTC_os.find('<spc>'), weight_one, final_space)
        for s in range(N ** 2):
            CTC.add_arc(s, final_space_arc)

    def get_input_sym(c, let):
        if let != 0 or use_contextual_blanks:
            # read label in context, shifted by eps
            return c * N + let + 1
        else:
            # read the global blank
            return 0 + 1

    for s1 in range(N ** 2):
        c1 = s1 // N
        l1 = s1 % N

        # emit the xy non-blank self loop of y in the context of x
        # please note that:
        # - xB and xx self loops will be handled later
        self_loop = None
        if allow_nonblank_selfloops and l1 != 0 and c1 != l1:
            # repeat the last symbol in its context
            in_symbol = get_input_sym(c1, l1)
            CTC.add_arc(s1, fst.Arc(in_symbol, os_eps, weight_one, s1))
            self_loop = s1

        # determine the next context
        if l1 == 0:  # last symbol read was a blank
            c2 = c1
        else:
            c2 = l1

        for l2 in range(N):
            s2 = c2 * N + l2
            # assert that we didn't emit the self loop before
            assert not self_loop == s2

            if (l2 == 0 or s1 == s2 or
                    (loop_using_symbol_repetitions and l1 == l2)):
                out_s = os_eps
            else:
                out_s = CTC_os.find(net_vocab[l2])

            CTC.add_arc(
                s1,
                fst.Arc(get_input_sym(c2, l2), out_s, weight_one, s2))

    CTC.arcsort('olabel')
    CTC.write(out_file)

def gen_trigram_graph(ngram_to_class_file, net_vocab_file, token_file, out_file,
                      add_final_space=False, use_contextual_blanks=False,
                      prevent_epsilons=False, determinize=True):

    net_vocab = read_net_vocab(net_vocab_file)
    print("net vocab", net_vocab)
    N = len(net_vocab)

    with open(ngram_to_class_file, 'r') as f:
        trigrams = [tuple([int(n) for n in line.split()]) for line in f]

    CTC = build_ctc_trigram_decoding_fst_v2(
        N, trigrams, arc_type='standard',
        use_context_blanks=use_contextual_blanks,
        prevent_epsilons=prevent_epsilons, determinize=determinize,
        add_syms=False)

    assert CTC.weight_type() == 'tropical'

    # Emitted symbols need to be remapped from net_vocab to token symbols
    #   net_vocab[:5] : ['<pad>', '<unk>', '<spc>', 'E', 'T']
    #   tokens[:5]    : ['<eps> 0', '<spc> 1', '<pad> 2', '<unk> 3', 'E 4']
    # <pad> is unused and gets mapped to eps, <unk> and <spc> change ids,
    # the rest is roughly shifted by 1.
    tokens = {t.split()[0] : int(t.split()[1]) for t in open(token_file, 'r')}
    net_vocab_dict = {t:i for i,t in enumerate(net_vocab)}
    osym_map = []
    for t,i in net_vocab_dict.items():
        osym_map.append((i, 0 if t == '<pad>' else tokens[t]))
    CTC.relabel_pairs(ipairs=None, opairs=osym_map)
    print(osym_map)

    CTC_os = fst.SymbolTable.read_text(token_file)
    CTC.set_output_symbols(CTC_os)
    os_eps = CTC_os.find('<eps>')
    assert os_eps == 0

    weight_one = fst.Weight.One('tropical')

    if add_final_space:
        is_final = lambda s: CTC.final(s) != fst.Weight(CTC.weight_type(),
                                                        'infinity')
        final_space = CTC.add_state()
        CTC.set_final(final_space)
        final_space_arc = fst.Arc(
            0, CTC_os.find('<spc>'), weight_one, final_space)
        for s in CTC.states():
            if is_final(s):
                CTC.add_arc(s, final_space_arc)

    CTC.arcsort('olabel')
    CTC.write(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fst_kind')
    parser.add_argument('out_file')
    parser.add_argument('--net_vocab')
    parser.add_argument('--tokens')
    parser.add_argument('--trigram_to_class_file')
    parser.add_argument('--add_final_space', action='store_true')
    parser.add_argument('--use_contextual_blanks', action='store_true')
    parser.add_argument('--no_allow_nonblank_selfloops',
                        action='store_true')
    parser.add_argument('--loop_using_symbol_repetitions', action='store_true')

    args = parser.parse_args()

    if args.fst_kind == 'ctc_unigram':
        gen_unigram_graph(
            args.net_vocab, args.tokens, args.out_file,
            args.add_final_space,
            allow_nonblank_selfloops=not args.no_allow_nonblank_selfloops,
            )
    elif args.fst_kind == 'ctc_bigram':
        gen_bigram_graph(
            args.net_vocab, args.tokens, args.out_file,
            args.add_final_space,
            allow_nonblank_selfloops=not args.no_allow_nonblank_selfloops,
            use_contextual_blanks=args.use_contextual_blanks,
            loop_using_symbol_repetitions=args.loop_using_symbol_repetitions,
            )
    elif args.fst_kind == 'ctc_trigram':

        if args.no_allow_nonblank_selfloops:
            raise ValueError('--no_allow_nonblank_selfloops '
                             'not supported in trigrams')
        if args.no_allow_nonblank_selfloops:
            raise ValueError('--loop_using_symbol_repetitions '
                             'not supported in trigrams')
        gen_trigram_graph(
            args.trigram_to_class_file,
            args.net_vocab, args.tokens, args.out_file,
            args.add_final_space,
            use_contextual_blanks=args.use_contextual_blanks,
            prevent_epsilons=False,
            determinize=False
            )
    else:
        raise ValueError('Unknown fst kind %s' % args.fst_kind)
