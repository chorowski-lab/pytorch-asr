from att_speech.modules.hooks.hook import TrainingLoopHook
import torch


class ConstantGradientNoise(TrainingLoopHook):
    def __init__(self, gradient_noise, **kwargs):
        self.gradient_noise = gradient_noise
        super(ConstantGradientNoise, self).__init__(**kwargs)

    def post_backward(self, model, optimizer, current_iteration, loss):
        var = self.gradient_noise / (1 + current_iteration)**0.55
        var = var**2
        for param_group in optimizer.param_groups:
            for parameter in param_group['params']:
                parameter.grad += (
                    torch.randn_like(parameter.grad) * var)
