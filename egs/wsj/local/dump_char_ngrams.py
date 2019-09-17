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


from att_speech.configuration import Configuration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', default=3, type=int)
    parser.add_argument('config')
    parser.add_argument('out_file')
    args = parser.parse_args()

    config = Configuration(args.config)
    train_data = config['Datasets']['train'].dataset

    order = args.order

    pad_left = order // 2
    pad_right = (order - 1) // 2
    assert order == pad_left + pad_right + 1

    ngrams = set()
    for _, text in train_data.texts:
        toks = train_data.tokenize(text, pad_left, pad_right)

        for i in range(len(toks) - order + 1):
            ngrams.add(tuple(toks[i:i+order]))

    with open(args.out_file, 'w') as of:
        for ngram in sorted(ngrams):
            of.write('%s\n' % ' '.join([str(i) for i in ngram]))
