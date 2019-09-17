from torch import nn
from abc import ABCMeta, abstractmethod

class BaseEncoder(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            raise RuntimeError(
                "Unrecognized options: {}".format(', '.join(kwargs.keys())))
        super(BaseEncoder, self).__init__()

    @abstractmethod
    def forward(self, features, features_lengths, spkids):
        """ Encode a minibatch of audio features

        :param features: float32 tensor of size (bs x t x f x c)
        :param features_lengths: int64 tensor of size (bs)
        :param spkids: string id of speakers
        :returns: A tuple with elements:
                    - encoded: float32 tensor of size (t x bs x d)
                    - encoded_lens: int64 tensor of size (bs)
        """
        pass

    def get_parameters_for_optimizer(self):
        return self.parameters()