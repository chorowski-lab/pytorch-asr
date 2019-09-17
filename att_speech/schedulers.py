#
# Jan Chorowski 2018, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class MultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    def step(self, metrics=None, epoch=-1):
        del metrics  # unused
        return super(MultiStepLR, self).step(epoch)
