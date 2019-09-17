#!/usr/bin/env python
#
# Jan Chorowski 2018, UWr
#
'''

'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import re
import sys


def apply_lexicon(lex, in_f, out_f):
    unk = lex['<UNK>']
    num_unk = 0
    for line in in_f:
        line = line.split()
        out = [line[0]]
        for w in line[1:]:
            if w[0] == w[-1] == '*':
                w = w[1:-1]
            try:
                out.extend(lex[w])
            except KeyError:
                out.extend(unk)
                num_unk += 1
        out_f.write(' '.join(out))
        out_f.write('\n')
    print("Wrote %s, %d unks" % (out_f.name, num_unk))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: %s check_vocab_file lexicon text1 text2..." % sys.argv[0])
        sys.exit(1)
    check_vocab_file = sys.argv[1]
    lex_file = sys.argv[2]

    lexicon = {'<pad>'}
    all_phones = set()
    del_digits = re.compile('[0-9]')
    for line in open(lex_file):
        line = line.strip()
        if not line:
            continue
        line = line.split()
        word = line[0]
        word_phones = [del_digits.sub('', ph) for ph in line[1:]]
        all_phones.update(word_phones)
        lexicon[word] = word_phones

    vocab_phones = set([ph.strip()
                        for ph in open(check_vocab_file) if ph.strip()])

    assert all_phones == vocab_phones
    for fname in sys.argv[3:]:
        with open(fname) as in_f:
            with open(fname + '_phn', 'w') as out_f:
                apply_lexicon(lexicon, in_f, out_f)
