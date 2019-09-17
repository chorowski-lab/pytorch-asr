import numpy as np
import random


class WavAugmenter(object):
    def __init__(self, dataset,
                 pitch_change=0, speed_change=0,
                 constant_pitch_change=0, constant_speed_change=0,
                 mix_another=None):
        self.dataset = dataset

        self.pitch_change = pitch_change
        self.speed_change = speed_change
        self.constant_pitch_change = constant_pitch_change
        self.constant_speed_change = constant_speed_change
        self.mix_another = mix_another

        self.passthrough = all(
            (self.pitch_change == 0,
             self.speed_change == 0,
             not self.mix_another,
             self.constant_pitch_change == 0,
             self.constant_speed_change == 0))
        self.do_assertions()

    def do_assertions(self):
        if self.passthrough:
            return

        assert all(f.strip().endswith('.wav') for f in self.dataset.inputs), \
            "All inputs must be wav's in order to use WavAugmenter"

        if self.mix_another:
            assert self.dataset.durs

    def _pitch_change(self):
        if (self.pitch_change == 0 and self.constant_pitch_change == 0):
            return ""

        if self.constant_pitch_change != 0:
            pitch_val = self.constant_pitch_change
        else:
            pitch_val = (
                random.random() * 2 * self.pitch_change -
                self.pitch_change)
        return "pitch %f " % (pitch_val,)

    def _speed_change(self):
        if self.speed_change == 0 and self.constant_speed_change == 0:
            return ""

        if self.constant_speed_change != 0:
            speed_val = self.constant_speed_change
        else:
            speed_val = (
                random.random() * 2 * self.speed_change -
                self.speed_change + 1.0)
        return "speed %f " % (speed_val,)

    def _mix_another(self, i):
        if not self.mix_another:
            return ""

        utt_id = self.dataset.uttids[i]

        snr_low, snr_high = np.log(self.mix_another)
        snr = np.exp(snr_low + random.random() * (snr_high - snr_low))
        other_uttid, other_wav = self.dataset.inputs[
            random.randrange(len(self.dataset.inputs))].split()
        dur = self.dataset.durs[utt_id]
        other_dur = self.dataset.durs[other_uttid]

        offset = random.random() * max(0, dur - other_dur)
        return (" | wav-reverberate --additive-signals='%s'"
                " --snrs='%f' --start-times='%f' - -" %
                (other_wav, snr, offset))

    def __call__(self, i):
        if self.passthrough:
            return self.dataset.inputs[i]

        ui, wav = self.dataset.inputs[i].split()

        new_inp = ("%s sox %s -t wav - %s %s %s|\n" %
                   (
                       ui,
                       wav,
                       self._speed_change(),
                       self._pitch_change(),
                       self._mix_another(i)
                    ))
        return new_inp
