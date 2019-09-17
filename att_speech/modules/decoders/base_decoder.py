# -*- coding: utf8 -*-
#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

from torch import nn


class BaseDecoder(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, vocabulary=None, **kwargs):
        super(BaseDecoder, self).__init__(**kwargs)
        if vocabulary is None:
            vocabulary = []
            self.num_symbols = None
        else:
            self.num_symbols = len(vocabulary)
        self.vocabulary = vocabulary + ['<eos>']

    @abstractmethod
    def forward(self, encoded, encoded_lens, texts, text_lens, **kwargs):
        """ (t x bs x d, lens) -> () """
        pass

    @abstractmethod
    def decode(self, encoded, encoded_lens, **kwargs):
        """  (t x bs x d, lens) -> bs string  """
        pass
