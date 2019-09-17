from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import os
import tempfile


import torch
import torch.utils.data

import numpy as np

import kaldi_io

#from att_speech.modules import ctc_losses
from .text_augmenter import TextAugmenter
from .wav_augmenter import WavAugmenter

from att_speech import fst_utils, utils


class KaldiDataset(torch.utils.data.Dataset):
    """
    KaldiDataset is subclass of torch.utils.data.Dataset for low-level
    interaction with Kaldi pipeline - reading one entry at a time
    """
    # pylint: disable=too-many-instance-attributes
    __metaclass__ = ABCMeta

    def __init__(self,
                 data_dir,
                 feat_pipeline="compute-fbank-feats %s %s 2>/dev/null",
                 feat_input_file="wav.scp",
                 durations_file="durs",
                 feat_dim=(-1, 23, 1),
                 feat_dim_shuffle=(0, 1, 2),
                 transforms=None,
                 max_text_length=None,
                 spkid_to_ivector_file=None,
                 space_symbol=None,
                 text_augmentation=None,
                 wav_augmentation=None,
                 text_file="text",
                 graph_generator=None):

        super(KaldiDataset, self).__init__()
        self.data_dir = data_dir
        self.feat_input_file = feat_input_file
        self.feat_pipeline = feat_pipeline
        self.feat_dim = feat_dim
        self.feat_dim_shuffle = feat_dim_shuffle
        self.uttid_to_spkid = {}
        self.uttid_to_text_len = {}
        self.uttids = []
        self.text_augmenter = TextAugmenter(
            text_augmentation, self.vocabulary.stoi.values(),
            self.vocabulary.stoi[space_symbol] if space_symbol else None)
        if not wav_augmentation:
            wav_augmentation = {}
        if graph_generator:
            self.graph_gen = utils.contruct_from_kwargs(
                graph_generator, 'att_speech.fst_utils',
                {'num_symbols': len(self.vocabulary.itos)})
        else:
            self.graph_gen = None

        texts = []
        with open(os.path.join(data_dir, text_file)) as text_f:
            for line in text_f:
                line = line.strip()
                if not line:
                    continue
                line = line.split(None, 1)
                if max_text_length and len(line[1]) > max_text_length:
                    continue
                self.uttids += [line[0]]
                texts += [line[1]]
                self.uttid_to_text_len[line[0]] = len(line[1])

        assert len(self.uttids) > 0, "Dataset cannot be empty"

        self.texts = zip(self.uttids, self.process_text(texts))
        self.uttids_map = dict(zip(self.uttids, range(len(self.uttids))))

        with open(os.path.join(data_dir, "utt2spk")) as text_f:
            for line in text_f:
                line = line.strip()
                if not line:
                    continue
                line = line.split(None, 1)
                uttid = line[0]
                if uttid in self.uttids_map:
                    self.uttid_to_spkid[line[0]] = line[1]
        assert len(self.uttid_to_spkid) == len(self.texts)

        self.inputs = []
        with open(os.path.join(data_dir, feat_input_file)) as input_f:
            for line in input_f:
                if len(self.inputs) == len(self.texts):
                    break
                line = line.strip()
                if not line:
                    continue
                uttid = line.split(None, 1)[0]
                if self.texts[len(self.inputs)][0] == uttid:
                    self.inputs.append(line + "\n")
        assert len(self.inputs) == len(self.texts)

        if not durations_file.startswith('/'):
            durs_fname = os.path.join(data_dir, durations_file)
        else:
            durs_fname = durations_file
        self.durs = None
        if os.path.exists(durs_fname):
            self.durs = {}
            with open(durs_fname) as durs_f:
                for line in durs_f:
                    if not line.strip():
                        continue
                    utt_id, dur = line.split()
                    self.durs[utt_id] = float(dur)

        # TODO: validating ivectors file
        # TODO: now spkid_to_ivector_file has to include full path to file
        if spkid_to_ivector_file is not None:
            self.spkid_to_ivector = kaldi_io.RandomAccessBaseFloatVectorReader(
                'scp:%s' % spkid_to_ivector_file)
        else:
            self.spkid_to_ivector = None

        self.transforms = transforms if transforms else []

        self.reader = None
        self.input_fifo = None

        self.wav_augmenter = WavAugmenter(self, **wav_augmentation)

    def num_classes(self):
        if self.graph_gen is not None:
            return self.graph_gen.num_classes
        return len(self.vocabulary.itos)

    def get_text_by_id(self, uttid):
        """
        Returns sentence text from utterance id
        """
        i = self.uttids_map[uttid]
        return self._decode_sentence(self.texts[i][1])

    @abstractmethod
    def _decode_sentence(self, sentence):
        """
        Returns string for given sentence
        :param sentence: sentence as seen in self.texts
        :returns: string
        """
        pass

    @abstractmethod
    def process_text(self, sentences):
        """
        Converts input text no normalized form
        :param sentences: list of strings
        :returns: normalized sentences, list of strings
        """
        pass

    @abstractmethod
    def tokenize(self, text):
        """
        Converts single sentence to int tokens
        :param text: sentence
        :returns: numpy vector of encoded sentence
        """
        pass

    def ids_to_chars_words_sentence(self, text_ids, ignore_noise=False):
        """Convert the ids to characters and words."""
        del ignore_noise  # unused
        # By default do the lookup in the vocab file
        itos = self.vocabulary.itos

        chars = [itos[id_] for id_ in text_ids]
        sentence = ''.join(chars)
        words = sentence.split()
        return chars, words, sentence

    def __len__(self):
        return len(self.texts)

    def create_reader(self):
        fifo_name = tempfile.mktemp()
        os.mkfifo(fifo_name)
        in_scp = "scp:" + fifo_name
        out_ark = "ark,f:-"
        feat_pipeline = self.feat_pipeline % (in_scp, out_ark)
        self.reader = kaldi_io.RandomAccessBaseFloatMatrixReader(
            "ark,o:%s |" % (feat_pipeline, ))
        self.input_fifo = open(fifo_name, "w")

    def __getitem__(self, i):
        utt_id, text = self.texts[i]
        text = self.text_augmenter(self.tokenize(text))
        if self.reader is None:
            self.create_reader()

        self.input_fifo.write(self.wav_augmenter(i))
        self.input_fifo.flush()

        feats = self.reader[utt_id]
        feats = feats.reshape(self.feat_dim)
        feats = feats.transpose(self.feat_dim_shuffle)
        # feats are now time  x feats x channels
        for transformer in self.transforms:
            feats = transformer(feats)

        spk_id = self.uttid_to_spkid[utt_id]
        ivectors = (None if self.spkid_to_ivector is None
                    else self.spkid_to_ivector[spk_id])

        ret = dict(uttid=utt_id,
                   text=text.astype('int32'),
                   spkid=self.uttid_to_spkid[utt_id],
                   feats=feats,
                   ivectors=ivectors)

        if self.graph_gen:
            ret['graph_matrices'] = self.graph_gen.get_training_matrices(
                ret['text'])

        return ret
