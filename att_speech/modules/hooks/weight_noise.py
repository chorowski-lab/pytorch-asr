from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from att_speech.modules.hooks.hook import TrainingLoopHook
import torch


class WeightNoise(TrainingLoopHook):
    def __init__(self, weight_noise, start_iteration,
                 modules_supporting_noise=None,
                 **kwargs):
        self.weight_noise = weight_noise
        self.start_iteration = start_iteration
        self.modules_supporting_noise = modules_supporting_noise or []
        self.rand_values = {}
        super(WeightNoise, self).__init__(**kwargs)

    def get_rand_val(self, name, current_iteration):
        raise NotImplementedError

    def get_base_weight_noise(self, name):
        if isinstance(self.weight_noise, dict):
            for k, v in self.weight_noise.iteritems():
                if name.startswith(k):
                    return v
            raise ValueError(
                    "No weight noise information for {}".format(name))
        else:
            return self.weight_noise

    def _requires_noise(self, weight_name):
        if 'weight' not in weight_name or 'batch_norm' in weight_name:
            return False
        for mod in self.modules_supporting_noise:
            if weight_name.startswith(mod):
                return False
        return True

    def _apply_module_attrs(self, model, current_iteration, reset=False):
        for mod_name in self.modules_supporting_noise:
            obj = model
            for field in mod_name.split('.'):
                obj = getattr(obj, field)
            assert hasattr(obj, 'weight_noise')
            if reset:
                noise_mag = 0.0
            else:
                noise_mag = self.get_rand_val(mod_name, current_iteration)
            # print("Mod noising ", mod_name, noise_mag)
            setattr(obj, 'weight_noise', noise_mag)

    def pre_train_forward(self, model, optimizer, current_iteration):
        if current_iteration > self.start_iteration:
            self.rand_values = {}
            self._apply_module_attrs(model, current_iteration)
            for name, weight in model.named_parameters():
                if self._requires_noise(name):
                    rand = (self.get_rand_val(name, current_iteration)
                            * torch.randn_like(weight))
                    self.rand_values[name] = rand
                    weight.data.add_(rand)

    def post_backward(self, model, optimizer, current_iteration, loss):
        if not self.rand_values:
            return
        self._apply_module_attrs(model, current_iteration, reset=True)
        for name, weight in model.named_parameters():
            if self._requires_noise(name):
                rand = self.rand_values[name]
                self.rand_values[name] = rand
                weight.data.add_(-rand)
        self.rand_values = {}


class ConstantWeightNoise(WeightNoise):
    def get_rand_val(self, name, current_iteration):
        return self.get_base_weight_noise(name)


class LinearIncreaseWeightNoise(WeightNoise):
    def pre_train_forward(self, model, optimizer, current_iteration):
        self.rand_values = {}
        self._apply_module_attrs(model, current_iteration)
        for name, weight in model.named_parameters():
            if self._requires_noise(name):
                noise_mag = self.get_rand_val(name, current_iteration)
                # print("Noising ", name, noise_mag)
                rand = torch.randn_like(weight)
                rand *= noise_mag
                self.rand_values[name] = rand
                weight.data.add_(rand)

    def get_rand_val(self, name, current_iteration):
        incr = min(
            1.0, float(current_iteration) / float(self.start_iteration))
        return incr * self.get_base_weight_noise(name)
