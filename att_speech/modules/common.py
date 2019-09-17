from torch import nn


class SequenceWise(nn.Module):
    """
    Collapses input of dim T*BS*F to (T*BS)*F, and applies to a module.
    Allows handling of variable sequence lengths and minibatch sizes.
    :param module: Module to apply input to.
    """
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        time, batch_size = x.size(0), x.size(1)
        x = x.view(time * batch_size, -1)
        x = self.module(x)
        x = x.view(time, batch_size, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr
