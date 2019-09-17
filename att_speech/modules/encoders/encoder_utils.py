from __future__ import print_function
from __future__ import division

from torch import nn
import torch


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Normalization(nn.Module):
    def __init__(self, norm_type, nary, input_size):
        super(Normalization, self).__init__()
        self.nary = nary
        if norm_type == 'batch_norm':
            if nary == 1:
                self.batch_norm = nn.BatchNorm1d(input_size)
            elif nary == 2:
                self.batch_norm = nn.BatchNorm2d(input_size)
            else:
                raise ValueError(
                    "Unknown nary for {} normalization".format(norm_type))
        elif norm_type == 'instance_norm':
            if nary == 1:
                self.batch_norm = nn.InstanceNorm1d(input_size)
            elif nary == 2:
                self.batch_norm = nn.InstanceNorm2d(input_size)
            else:
                raise ValueError(
                    "Unknown nary for {} normalization".format(norm_type))
        elif not norm_type or norm_type == 'none':
            self.batch_norm = Identity()
        else:
            raise ValueError(
                """Unknown normalization type {}.
                   Possible are: batch_norm, instance_norm or none"""
                .format(norm_type))

    def forward(self, x, speaker=None):
        if self.nary >= 2:
            return self.batch_norm(x)
        elif self.nary == 1:
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x_data = self.batch_norm(x.data)
                return torch.nn.utils.rnn.PackedSequence(x_data, x.batch_sizes)
            else:
                return self.batch_norm(x)


class BatchRNN(nn.Module):
    """
    RNN with normalization applied between layers.
    :param input_size: Size of the input
    :param hidden_size: Size of hidden state
    :param rnn_type: Class for initializing RNN
    :param bidirectional: is it bidirectional
    :param packed_data: Will input to the module be packed
                        with pack_padded_sequence
    :param normalization: String, what type of normalization to use.
                          Possible options: 'batch_norm', 'instance_norm',
                          'per_speaker_norm', 'per_speaker_global_affine_norm'
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM,
                 bidirectional=False, packed_data=False, normalization=None,
                 projection_size=0, residual=False, subsample=False):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.residual = residual
        self.batch_norm = Normalization(normalization, 1, input_size)

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

        self.subsample = subsample

        if projection_size > 0:
            self.projection = torch.nn.Linear(
                hidden_size * self.num_directions, projection_size,
                bias=False)
        else:
            self.projection = None

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    """
    :input x: input of size t x bs x f
    """
    def forward(self, x, speakers):
        if self.residual:
            res = x.data
        x = self.batch_norm(x, speakers)
        x, _ = self.rnn(x)
        if self.subsample:
            x, lengths = nn.utils.rnn.pad_packed_sequence(x)
            x = x[::2]
            x = nn.utils.rnn.pack_padded_sequence(x, lengths // 2)
        if self.projection:
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x_data = self.projection(x.data)
                x = torch.nn.utils.rnn.PackedSequence(x_data, x.batch_sizes)
            else:
                x = self.projection(x)
        elif self.bidirectional:
            # (TxBSxH*2) -> (TxBSxH) by sum
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x_data = (x.data.view(x.data.size(0), 2, -1)
                           .sum(1).view(x.data.size(0), -1))
                x = torch.nn.utils.rnn.PackedSequence(x_data, x.batch_sizes)
            else:
                x = (x.view(x.size(0), x.size(1), 2, -1)
                      .sum(2).view(x.size(0), x.size(1), -1))
        if self.residual:
            x_data = torch.nn.functional.relu(x.data + res)
            x = torch.nn.utils.rnn.PackedSequence(x_data, x.batch_sizes)
        return x


class SequentialWithOptionalAttributes(nn.Sequential):
    def forward(self, input, *args):
        for module in self._modules.values():
            params_count = module.forward.func_code.co_argcount
            # params_count is self + input + ...
            input = module(input, *args[:(params_count-2)])
        return input
