from att_speech.modules.hooks.hook import TrainingLoopHook
import torch

INF = float('inf')
MINF = float('-inf')


class KillOnNan(TrainingLoopHook):
    def __init__(self, *args, **kwargs):
        super(KillOnNan, self).__init__(*args, **kwargs)
        self.grace_init_val = 10
        self.grace_counter = self.grace_init_val

    def pre_backward(self, model, optimizer, current_iteration, loss):
        skip_step = 0
        if torch.isnan(loss).item():
            print('Loss is nan. Killing soon...')
            skip_step = 1
        elif loss.item() == INF or loss.item() == MINF:
            print('Loss is inf. Killing soon...')
            skip_step = 1
        self.grace_counter -= skip_step
        if self.grace_counter <= 0:
            print('Loss was nan/inf too many times. Killing.')
            exit(1)
        return skip_step == 1
