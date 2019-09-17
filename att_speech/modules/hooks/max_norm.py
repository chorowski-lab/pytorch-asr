from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from att_speech.modules.hooks.hook import TrainingLoopHook
import torch


class MaxNorm(TrainingLoopHook):
    def __init__(self, magnitude, tgt_modules, **kwargs):
        self.magnitude = magnitude
        self.tgt_modules = tgt_modules
        super(MaxNorm, self).__init__(**kwargs)

    def _requires_norm(self, weight_name):
        if not weight_name.endswith('weight') or 'batch_norm' in weight_name:
            return False
        for mod in self.tgt_modules:
            if weight_name.startswith(mod):
                return True
        return False

    def post_backward(self, model, optimizer, current_iteration, loss):
        for name, weight in model.named_parameters():
            if self._requires_norm(name):
                print(name)
                scale = self.magnitude / torch.max(torch.norm(weight, dim=1), dim=0)[0]
                # import IPython
                # IPython.embed()
                if scale < 1.0:
                    print("Applying scale %f to %s" % (scale.item(), name))
                    weight.data.mul_(scale)
