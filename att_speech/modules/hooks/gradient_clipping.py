from __future__ import division

import numpy as np

from torch.nn.utils import clip_grad_norm_
from att_speech.modules.hooks.hook import TrainingLoopHook
from att_speech.logger import DefaultTensorLogger, TensorLogger


logger = DefaultTensorLogger()


class GradientClipping(TrainingLoopHook):
    def __init__(self, clip_norm, skip_step_norm=np.inf, **kwargs):
        self.clip_norm = clip_norm
        self.skip_step_norm = skip_step_norm
        self.gstats = None
        super(GradientClipping, self).__init__(**kwargs)

    def post_backward(self, model, optimizer, current_iteration, loss):
        unclipped_norm = clip_grad_norm_(
            model.get_parameters_for_optimizer(), self.clip_norm)
        clipped = int(unclipped_norm > self.clip_norm)
        skipped = int(unclipped_norm > self.skip_step_norm)

        if self.gstats is None:
            self.gstats = (
                1,
                unclipped_norm, unclipped_norm, unclipped_norm,
                clipped, skipped)
        else:
            (acc_steps, g_min, g_sum, g_max, n_clipped, n_skipped
             ) = self.gstats
            self.gstats = (
                acc_steps + 1,
                min(g_min, unclipped_norm),
                g_sum + unclipped_norm,
                max(g_max, unclipped_norm),
                n_clipped + clipped,
                n_skipped + skipped)
        if logger.is_currently_logging():
            (acc_steps, g_min, g_sum, g_max, n_clipped, n_skipped
             ) = self.gstats
            logger.log_scalar("gclip/min", g_min)
            logger.log_scalar("gclip/max", g_max)
            logger.log_scalar("gclip/mean", 1.0 * g_sum / acc_steps)
            logger.log_scalar("gclip/clipfrac", 1.0 * n_clipped / acc_steps)
            logger.log_scalar("gclip/skipfrac", 1.0 * n_skipped / acc_steps)
            self.gstats = None
        if clipped:
            print("Grad clipped by ", 1.0 * self.clip_norm / unclipped_norm)
        # Tell the trainer if we should skip this step
        return bool(skipped)
