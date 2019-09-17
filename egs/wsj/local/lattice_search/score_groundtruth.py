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
from att_speech import fst_utils


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

def score_acoustic(dataset, model, item, sent):
    #sent = [dataset.dataset.vocabulary.stoi[c] for c in sent]
    encoded = model.encoder(item['features'][0].cuda(), item['features'][1], item['spkids'], None)
    #batch_size = encoded[0].size()[1]
    enc_state = model.decoder.enc_initial_state(encoded[0], encoded[1], 1, 1)
    my_logits = []
    sent = sent + [len(dataset.dataset.vocabulary.itos)]
    coverage = enc_state['att_weights'].detach()
    for i in range(len(sent)):
        prev_inputs = enc_state['inputs']
        logits, enc_state = model.decoder.enc_step(**enc_state)
        coverage += enc_state['att_weights'].detach()
        logprobs = torch.nn.functional.log_softmax(logits[0][0])
        my_logits += [ logprobs[sent[i]].item() ]
        new_input = torch.LongTensor([sent[i]])
        new_input = new_input.to(encoded[0].device)
        enc_state['inputs'] = torch.cat(
            (prev_inputs[1:], model.decoder.embedding(new_input).unsqueeze(0)))

    coverages = (
        (coverage > model.decoder.coverage_tau).sum(dim=0).float())
    coverage_scores = model.decoder.coverage_weight * coverages

    return sum(my_logits), coverage_scores.item()

def score_lm(dataset, model, sent):
    #sent = [dataset.dataset.vocabulary.stoi[c] for c in sent]
    nodes = {model.decoder.lm.start(): 0}
    for s in sent + [49]: # +eos
        nodes = fst_utils.expand(model.decoder.lm, nodes, model.decoder.alphabet_mapping[s], use_log_probs=True)
    return fst_utils.reduce_weights(nodes.values(), True) * -0.8

def rescore2(dataset, model, item, sent):
    model.decoder.beam_size = 1
    acoustic, coverage = score_acoustic(dataset, model, item, sent)
    lm = score_lm(dataset, model, sent)
    return {
        'acoustic': acoustic,
        'coverage': coverage,
        'lm': lm,
        'loss': (acoustic + coverage + lm) / (len(sent) ** model.decoder.length_normalization)
    }

def main():
    parser = get_parser()
    args = parser.parse_args()
    Globals.cuda = args.cuda

    subset = args.subset
    csv = args.csv

    config, model = get_config_and_model(args)

    if Globals.cuda:
        model.cuda()

    dataset = config['Datasets'][subset]

    with open(csv, 'w') as f:
        f.write('uttid,sentence,acoustic,lm,coverage,loss\n')
        for batch in dataset:
            sent = batch['texts'][0][0].tolist()
            utt = batch['uttids'][0]
            val = rescore2(dataset, model, batch, sent)
            sent_s = ''.join([dataset.dataset.vocabulary.itos[i] for i in sent])
            f.write(",".join(map(str, (utt, sent_s, val['acoustic'], val['lm'], val['coverage'], val['loss']))))
            f.write('\n')
            print(sent_s)

if __name__ == "__main__":
    main()
