# -*- coding: utf8 -*-
#
# Jan Chorowski 2017, UWr
#
'''

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import torch

from att_speech.configuration import Configuration, Globals
from att_speech.utils import extract_modify_dict
from att_speech.model_utils import latest_checkpoint


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument('save_dir', help="Directory to save all model files")
    parser.add_argument('remote_save_dir', nargs='?',
                        help="Remote directory to synchronize results to")
    parser.add_argument('-c', '--continue-training',
                        help='Continue experiment from given checkpoint')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Run ptvsd debugging server, and wait for attach')
    parser.add_argument('-m', '--modify_config', nargs='+',
                        help="List of config modifications")
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='use CUDA')
    parser.add_argument('--rng_seed', default=None, type=int,
                        help='reset the rng seed')
    parser.add_argument('--initialize-from', default=None,
                        help='Load weights from')
    parser.add_argument('--decode-first', action='store_true',
                        help='Run decoding before the training')
    parser.add_argument('--debug-skip-training', action='store_true',
                        help='For debugging finish training after 1 mnibatch')
    return parser


def get_config_filename(save_dir):
    template = 'train_config{}.yaml'
    if os.path.isfile(os.path.join(save_dir, template.format(''))):
        return os.path.join(save_dir, template.format(''))
    else:
        i = 1
        while os.path.isfile(os.path.join(save_dir, template.format(i))):
            i += 1
        return os.path.join(save_dir, template.format(i))


def initialize_from(model, path):
    state_dict = torch.load(path)['state_dict']
    model_dict = model.state_dict()

    print("Initializing parameters from {}:".format(path))
    loaded = []
    for k in sorted(model_dict.keys()):
        if k in state_dict:
            param = state_dict[k]
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            model_dict[k].copy_(param)
            loaded.append(k)
    print("Loaded: {}".format(loaded))
    print("Missing: {}".format(
        sorted(set(model_dict.keys()) - set(state_dict.keys()))))
    print("Unknown: {}".format(
        sorted(set(state_dict.keys()) - set(model_dict.keys()))))


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
        print('Wait for attach')
        ptvsd.wait_for_attach()
        print('Attached')

    if args.rng_seed is not None:
        print("Reseting the random seed")
        torch.manual_seed(args.rng_seed)

    Globals.cuda = args.cuda
    modify_dict = extract_modify_dict(args.modify_config)
    config = Configuration(args.config, modify_dict,
                           get_config_filename(args.save_dir))

    train_data = config['Datasets']['train']
    eval_data = {key: config['Datasets'][key]
                 for key in config['Datasets'].keys() if key != 'train'}
    model = config['Model']

    if args.initialize_from:
        initialize_from(model, args.initialize_from)

    print("Model summary:\n%s" % (model,))
    print("Model params:\n%s" % ("\n".join(
        ["%s: %s" % (p[0], p[1].size()) for p in model.named_parameters()])))
    print("Start training")
    trainer = config['Trainer']

    saved_state = None
    if args.continue_training == 'LAST':
        args.continue_training = latest_checkpoint(args.save_dir)
    if args.continue_training is not None:
        print('Loading state from...', args.continue_training)
        saved_state = torch.load(args.continue_training)

    trainer.run(args.save_dir, model, train_data, eval_data,
                saved_state=saved_state,
                debug_skip_training=args.debug_skip_training,
                decode_first=args.decode_first)


if __name__ == "__main__":
    main()
