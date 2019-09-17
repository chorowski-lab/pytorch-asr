# -*- coding: utf8 -*-
#
# Michał Zapotoczny 2017, UWr
#
'''

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from att_speech.configuration import Configuration, Globals
from att_speech.utils import evaluate_greedy, extract_modify_dict
from att_speech.model_utils import get_config_and_model


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
    parser.add_argument("--polyak", help="Use Polyak averaged model",
                        default=None, type=float)
    parser.add_argument("--model", nargs="?", help="Path to the model")
    parser.add_argument("--csv", nargs="?", help="Output to the csv file")
    parser.add_argument("--with-data-losses", action="store_true",
                        help="""Report loss of every original and recognized
                              sentence in csv file""")
    parser.add_argument("--subset", help="Which subset to use (test or dev)",
                        default="test")
    parser.add_argument('-m', '--modify_config', nargs='+',
                        help="List of config modifications")
    parser.add_argument('--cuda', default=torch.cuda.is_available(),
                        help='use CUDA', type=str2bool)
    parser.add_argument('--no-strict', action='store_true',
                        help="allow unknown params in pickles")
    return parser


def evaluate_to_file(dataset, model, output_file, subset_name,
                     progress_callback=None):
    # output_file.write('Id;Recognized sentence;Original sentence;CER;WER\n')
    # output_file.flush()

    header = [
        'Id', 'Recognized', 'Original',
        '{} CER'.format(subset_name), '{} WER'.format(subset_name)]
    header += ['CER INS', 'CER DEL', 'CER SUB']
    header += ['WER INS', 'WER DEL', 'WER SUB']

    headers = {'data': [], 'found': False}

    def callback(uttid, recognized, original, cer, cer_stat, wer, wer_stat, other, **kwargs):
        if not headers['found']:
            additional_header = other.keys()
            output_file.write(
                ';'.join(header + [h + ' loss' for h in additional_header]))
            output_file.write('\n')
            headers['data'] = additional_header
            headers['found'] = True
        output = [uttid, recognized, original, cer, wer]
        for t in ['ins', 'del', 'sub']:
            output += [cer_stat[t]]
        for t in ['ins', 'del', 'sub']:
            output += [wer_stat[t]]
        output += [other[h] for h in headers['data']]
        output_file.write(';'.join(str(o) for o in output))
        output_file.write('\n')

    result = evaluate_greedy(dataset,
                             model,
                             callback,
                             progress_callback,
                             generate_data_losses=False)
    output_file.flush()
    return result


def main():
    parser = get_parser()
    args = parser.parse_args()
    Globals.cuda = args.cuda

    subset = args.subset

    config, model = get_config_and_model(args)

    if Globals.cuda:
        model.cuda()

    def progress_clb(*x):
        print("Processing batch {}/{} ({} elements)".format(*x))

    if args.csv:
        with open(args.csv, 'w', 1) as output_file:
            print(evaluate_to_file(
                config['Datasets'][subset], model, output_file, subset, progress_clb))
    else:
        print(evaluate_greedy(config['Datasets'][subset], model,
                              progress_callback=progress_clb))


if __name__ == "__main__":
    main()
