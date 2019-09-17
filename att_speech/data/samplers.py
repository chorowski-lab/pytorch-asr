# -*- coding: utf8 -*-
#
# Jan Chorowski 2017, UWr
#
'''
Dataset module
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data


class SortByLengthSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices_per_length = {}
        for uttid, text_len in self.data_source.uttid_to_text_len.iteritems():
            self.indices_per_length[text_len] = (
                    self.indices_per_length.get(text_len, [])
                    + [data_source.uttids_map[uttid]])
        self.sorted_by_length = []
        for k in sorted(self.indices_per_length.keys()):
            if k > 100:
                break
            self.sorted_by_length += self.indices_per_length[k]

    def __iter__(self):
        return iter(self.sorted_by_length)

    def __len__(self):
        return len(self.sorted_by_length)
