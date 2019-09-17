# -*- coding: utf8 -*-
#
# Michał Zapotoczny 2017, UWr
#
'''
WSJ dataset specification

Exemplary usage:
In [1]: from egs.wsj import data

In [2]: train_dataset = data.WSJDataset(data_dir="exp/wsjs5/data/train_si284/")

In [3]: train_dataset[10]
compute-fbank-feats scp:/pio/tmp/jch/tmpIIdEfu ark,f:-
Out[3]:
{'feats': Shape: (924, 23)
 [[  5.71403503   5.38562536   5.71528482 ...,  13.09023476  13.43752193
    13.45565796]
  [  5.96805573   5.56060362   6.76619244 ...,  12.91015339  13.35514259
    13.41080093]
  [  4.35371304   6.47779226   6.71086788 ...,  13.17774963  13.10784721
    12.59034729]
  ...,
  [  4.27286243   4.67225122   5.6490531  ...,  13.17068672  13.39624023
    13.29343987]
  [  4.84718466   6.09786224   6.66720581 ...,  13.17766476  13.01885986
    13.40524197]
  [  4.62433052   6.21121645   6.37919617 ...,  13.2691927   13.38753796
    13.17092609]], 'id': '011c020b', 'text': Shape: (136,)
 [ 8  4  9  2 11  8 19 11  3 10  2 14  6  9  4  2  9  8  9  4  3 10  2  3  5
   9  4  3 10  7  2  5  8 10 12  8  7  3  9  2 11  5  9  2  9 11  6 22  7  2
   9  8 19  7  9  2  6 18  2 18  8  7  5  7 14  8  5 12  2 17 10  6 19 10  3
   9  9  2 13  3  9 17  8  4  3  2 14  6  7  4  8  7 15  8  7 19  2 14  6  7
  18 10  6  7  4  5  4  8  6  7  9  2 22  8  4 11  2  8  4  9  2  3 16 17 12
   6 20  3  3  2 15  7  8  6  7  9]}

In [4]: dev_datset = data.WSJDataset(data_dir="exp/wsjs5/data/test_dev93/",
                                     copy_vocab_from=train_dataset)

In [5]: test_datset = data.WSJDataset(data_dir="exp/wsjs5/data/test_eval92/",
                                      copy_vocab_from=train_dataset)

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

import numpy as np

from att_speech.data import KaldiDataset
from egs.vocab import Vocab


class WSJDataset(KaldiDataset):
    """
    A WSJ Dataset concrete KaldiDataset class
    """
    def __init__(self, vocab_file=None, split_by_space=False,
                 copy_vocab_from=None, extra_eos=None,
                 cleanup_angle_brackets=False, **kwargs):
        assert copy_vocab_from is None
        if copy_vocab_from is not None and vocab_file is not None:
            raise ValueError(
                "Specify either copy_vocab_from to copy the vocabulary "
                "from the other dataset, or vocab_file to load "
                "for file")
        if copy_vocab_from is not None:
            self.vocabulary = copy_vocab_from.vocabulary
        elif vocab_file:
            self.vocabulary = Vocab.from_file(vocab_file)
        self.split_by_space = split_by_space
        self.extra_eos = extra_eos
        self.cleanup_angle_brackets = cleanup_angle_brackets
        super(WSJDataset, self).__init__(**kwargs)

    def tokenize(self, text, pad_left=0, pad_right=0):
        """
        Tokenize text by replacing each letter with integer from vocabulary
        """
        stoi = self.vocabulary.stoi
        if self.split_by_space:
            assert pad_left == pad_right == 0
            out_text = np.array(
                [stoi[phone] for phone in text.split()],
                dtype=np.int)
        else:
            out_text = np.zeros((len(text) + pad_left + pad_right,),
                                dtype=np.int)
            for i, char in enumerate(text):
                out_text[i + pad_left] = self.vocabulary.stoi[char]
        return out_text

    def ids_to_chars_words_sentence(self, text_ids, ignore_noise=False):
        if ignore_noise:
            text_ids = (_id for _id in text_ids if _id != self.vocabulary.stoi['~'])
        return super(WSJDataset, self).ids_to_chars_words_sentence(text_ids)

    def process_text(self, sentences):
        """
        Character data normalization

        Data cleaning remarks:
        Two utterances have empty transaltion ~~
        *text* means insertion
        ` occurs two times instead of '
        < and > occure only two times aoutside of the <NOISE> phrase

        We: replace <NOISE> with ~
        Then we get the list of unique characters and encode space as <spc>
        and ~ again as <NOISE> thus the two empty sequences are spelled out
        as <NOISE> <NOISE>
        """
        result = []
        for sentence in sentences:
            sentence = sentence\
                    .replace('<NOISE>', '~')\
                    .replace('`', "'")
            if self.cleanup_angle_brackets:
                sentence = sentence.replace('<', '').replace('>', '')
            if self.extra_eos is not None:
                sentence += self.extra_eos
            result += [sentence]

        return result

    def _decode_sentence(self, sentence):
        """
        Trivial one
        """
        return sentence


class WSJBigramDataset(WSJDataset):
    """
    WSJ in which each token is a fixed biphone
    """
    def __init__(self, **kwargs):
        self.split_by_space = False
        super(WSJBigramDataset, self).__init__(**kwargs)

    def num_classes(self):
        return len(self.vocabulary.itos) ** 2

    def tokenize(self, text):
        """
        Tokenize text by replacing each letter with integer from vocabulary
        """
        stoi = self.vocabulary.stoi

        num_phones = len(stoi)
        if self.split_by_space:
            text = text.split()
        out_text = np.empty((len(text,)), dtype=np.int)
        last_p = 0
        for i, p in enumerate(text):
            p = stoi[p]
            out_text[i] = last_p * num_phones + p
            last_p = p
        return out_text

    def ids_to_chars_words_sentence(self, text_ids, ignore_noise=False):
        """Convert the ids to characters and words."""
        # By default do the lookup in the vocab file
        itos = self.vocabulary.itos

        chars = [itos[id_ % len(itos)] for id_ in text_ids]
        if ignore_noise:
            chars = [c for c in chars if c != '~']
        sentence = ''.join(chars)
        words = sentence.split()
        return chars, words, sentence


class WSJNgramDataset(WSJDataset):
    """
    WSJ in which each token is a character ngram
    """
    def __init__(self, ngrams, sloppy=False, **kwargs):
        self.split_by_space = False
        self.sloppy = sloppy
        self.id_to_ngram = []
        self.ngram_to_id = {}
        self.order = None
        with open(ngrams) as f:
            for line in f:
                ngram = tuple((int(i) for i in line.split()))
                if not self.order:
                    self.order = len(ngram)
                else:
                    assert self.order == len(ngram)
                self.id_to_ngram.append(ngram)
                self.ngram_to_id[ngram] = len(self.ngram_to_id)
        self.pad_left = self.order // 2
        self.pad_right = (self.order - 1) // 2

        super(WSJNgramDataset, self).__init__(**kwargs)

    def num_classes(self):
        return len(self.id_to_ngram)

    def tokenize(self, text, pad_left=0, pad_right=0):
        """
        Tokenize text by replacing each letter with integer from vocabulary
        """
        assert pad_left == pad_right == 0
        toks = WSJDataset.tokenize(
            self, text, self.pad_left, self.pad_right)
        order = self.order
        ngram_toks = np.zeros((len(toks) - order + 1), dtype=np.int32)
        for i in range(len(ngram_toks)):
            if not self.sloppy:
                ngram_toks[i] = self.ngram_to_id[tuple(toks[i:i+order])]
            else:
                ngram_toks[i] = self.ngram_to_id.get(
                    tuple(toks[i:i+order]), 0)
        return ngram_toks

    def ids_to_chars_words_sentence(self, text_ids, ignore_noise=False):
        """Convert the ids to characters and words."""
        # By default do the lookup in the vocab file
        itos = self.vocabulary.itos
        iton = self.id_to_ngram
        middle_i = self.order // 2
        chars = [itos[iton[id_][middle_i]] for id_ in text_ids]
        if ignore_noise:
            chars = [c for c in chars if c != '~']
        sentence = ''.join(chars)
        words = sentence.split()
        return chars, words, sentence
