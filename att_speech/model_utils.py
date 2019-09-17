from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import sys

import torch

from att_speech import configuration, utils


def glob_latest(pattern):
    pl, pr = pattern.split('*')
    files = glob.glob(pattern)
    if not files:
        return None

    def key(x):
        return int(x[len(pl):-len(pr)])
    return max(files, key=key)


def glob_first(pattern):
    files = glob.glob(pattern)
    if not files:
        return None

    return files[0]


def best_checkpoint(save_dir):
    t = glob_first(os.path.join(
        save_dir, 'checkpoints', '*CER*.pkl'))
    if not t:
        t = glob_first(os.path.join(
            save_dir, 'checkpoints', '*cer*.pkl'))
    return t


def latest_checkpoint(save_dir):
    return glob_latest(os.path.join(
        save_dir, 'checkpoints', 'checkpoint_*.pkl'))


def latest_config(save_dir):
    return glob_latest(os.path.join(
        save_dir, 'train_config*.yaml'))


def get_config_and_model(args):
    configuration.Globals.cuda = args.cuda
    modify_dict = utils.extract_modify_dict(args.modify_config)

    if os.path.isdir(args.config):
        args.config = latest_config(args.config)
        if not args.config:
            sys.stderr.write("--config is a dir, but it does not contian"
                             "train_config*.yaml files.\n")

    if not args.model:
        args.model = latest_checkpoint(os.path.dirname(args.config))
    elif args.model.lower() == 'best':
        args.model = best_checkpoint(os.path.dirname(args.config))

    sys.stderr.write("Loading config from %s\n" % (args.config))
    config = configuration.Configuration(args.config, modify_dict)

    model = config['Model']

    sys.stderr.write("Loading saved model from %s\n" % (args.model))

    ########
    # load model. Should be replaced with right code.
    if configuration.Globals.cuda:
        state_dict = torch.load(args.model)
    else:
        state_dict = torch.load(args.model,
                                map_location=lambda storage, _: storage)
    state_dict_name = 'state_dict'
    if args.polyak:
        state_dict_name = 'avg_state_dict_%f' % (args.polyak,)
        if state_dict_name not in state_dict:
            state_dict_name = max(
                [k for k in state_dict if k.startswith('avg_state_dict')],
                key=utils.natural_keys)
    print("Loading state dict %s" % (state_dict_name,))
    model.load_state(state_dict[state_dict_name], strict=(not args.no_strict))
    #######

    if configuration.Globals.cuda:
        model.cuda()
    model.eval()

    return config, model
