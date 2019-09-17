# -*- coding: utf8 -*-
#
# Micha?? Zapotoczny 2017, UWr
#
'''

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np

import torch

from torch.autograd import Variable

import kaldi_io

from att_speech.configuration import Globals
from att_speech.model_utils import get_config_and_model

EPSILON = 1e-30


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument("out_wspec", help="Kaldi logits output")
    parser.add_argument("--model", nargs="?", help="Path to the model")
    parser.add_argument("--subset", help="Which subset to use (test or dev)",
                        default="test")
    parser.add_argument("--polyak", help="Use Polyak averaged model",
                        default=None, type=float)
    parser.add_argument('-m', '--modify_config', nargs='+',
                        help="List of config modifications")
    parser.add_argument('--cuda', default=torch.cuda.is_available(),
                        help='use CUDA', type=str2bool)
    parser.add_argument('--transfer-hash-prob', action='store_true',
                        help='transfer probability mass from hash to blank')
    parser.add_argument('--imitate-biphones', action='store_true',
                        help=('replicate outputs to simulate biphone outputs '
                              'of monophone model'))
    parser.add_argument('--block-normalize', action='store_true',
                        help=('force normalization of outputs '
                              'in biphone blocks'))
    parser.add_argument('--block-marginalize', action='store_true',
                        help=("marginalize probabilities over contexts; "
                              "return S instead of S**2 probs per frame"))
    parser.add_argument('--no-strict', action='store_true',
                        help="allow unknown params in pickles")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    config, model = get_config_and_model(args)

    dataset = config['Datasets'][args.subset]
    # Remove the training graph generator from the dataset. The testing
    # data may have characters for which we don't have CD symbols
    # and the graphs are not needed to compute the logits
    dataset.dataset.graph_gen = None
    owriter = kaldi_io.BaseFloatMatrixWriter(args.out_wspec)

    for j, batch in enumerate(dataset):
        sys.stderr.write("Processing batch %d/%d\n" % (j + 1, len(dataset)))
        feature_lens = Variable(batch['features'][1])
        features = Variable(batch['features'][0])
        speakers = batch['spkids']

        if Globals.cuda:
            features = features.cuda()

        with torch.no_grad():
            encoded, encoded_lens = model.encoder(
                features, feature_lens, speakers)
            # t x bsz x num_classes
            logprobs = model.decoder.logits(encoded, encoded_lens)
            logprobs = logprobs.data.cpu().numpy()

        # transfer probability mass from hash `#` to blank `<pad>`
        if args.transfer_hash_prob:
            blank_probs = np.exp(logprobs[:, :, 0])
            hash_probs = np.exp(logprobs[:, :, 3])
            blank_probs += hash_probs - EPSILON
            hash_probs = EPSILON
            logprobs[:, :, 0] = np.log(blank_probs)
            logprobs[:, :, 3] = np.log(hash_probs)

        t, bsz, num_classes = logprobs.shape

        if args.imitate_biphones:
            logprobs = np.tile(logprobs, (1, 1, num_classes))
            num_classes = num_classes ** 2
            if not args.block_normalize:
                num_mono = int(np.round(num_classes ** 0.5))
                z = np.exp(logprobs).sum(axis=2, keepdims=True)
                # This epsilon has to be really tiny,
                # otherwise not normalizes properly
                logprobs -= np.log(z + EPSILON)
        elif args.block_normalize:
                num_mono = int(np.round(num_classes ** 0.5))
                z = np.exp(logprobs).reshape(t, bsz, num_mono, num_mono)
                z = z.sum(axis=3).repeat(num_mono, axis=2)
                logprobs -= np.log(z + EPSILON)
        elif args.block_marginalize:
            print("Block-marginalizing probabilities.")
            num_symbols = int(np.round(num_classes ** 0.5))
            probs = np.exp(logprobs)
            probs = (probs.reshape(t, bsz, num_symbols, num_symbols)
                     .sum(axis=2) / num_symbols)
            logprobs = np.log(probs)
            assert not np.any(np.isnan(logprobs))
        for i in np.argsort(batch['uttids']):
            example_len = encoded_lens[i]
            owriter[batch['uttids'][i]] = logprobs[:example_len, i, :]


if __name__ == "__main__":
    sys.stderr.write("%s %s\n" % (os.path.basename(__file__), sys.argv))
    main()
