import collections
from tensorboardX import SummaryWriter


class TensorLogger(object):
    NO_LOG_STATE, NULL_LOG_STATE, STEP_LOG_STATE = (
            'NO_LOG', 'NULL_LOG', 'STEP_LOG')

    # global cache of sumary writetrs
    summary_writers = {}

    def __init__(self):
        self.log_state = self.NO_LOG_STATE
        self.summary_writer = None
        self.iteration = None

    def get_summary_writer(self, path):
        if path not in self.summary_writers:
            self.summary_writers[path] = SummaryWriter(path)
        return self.summary_writers[path]

    def ensure_no_log(self):
        if self.log_state != self.NO_LOG_STATE:
            raise Exception("Cannot call function during active step")

    def ensure_during_log(self):
        if self.log_state == self.NO_LOG_STATE:
            raise Exception("Cannot call function without activating step")

    def is_currently_logging(self):
        self.ensure_during_log()
        return self.log_state == self.STEP_LOG_STATE

    def make_step_log(self, log_dir, iteration):
        self.ensure_no_log()
        self.log_state = self.STEP_LOG_STATE
        self.iteration = iteration
        self.summary_writer = self.get_summary_writer(log_dir)

    def make_null_log(self):
        self.ensure_no_log()
        self.log_state = self.NULL_LOG_STATE

    def end_log(self):
        self.ensure_during_log()
        if self.log_state == self.STEP_LOG_STATE:
            for writer in self.summary_writer.all_writers.values():
                writer.flush()
            self.summary_writer = None
            self.iteration = None
        self.log_state = self.NO_LOG_STATE

    def log_scalar(self, name, value):
        if self.is_currently_logging():
            self.summary_writer.add_scalar(name, value, self.iteration)

    def log_histogram(self, name, values):
        if self.is_currently_logging():
            self.summary_writer.add_histogram(name, values, self.iteration)

    def log_image(self, tag, img):
        if self.is_currently_logging():
            self.summary_writer.add_image(tag, img, self.iteration)

    def log_audio(self, tag, audio):
        if self.is_currently_logging():
            self.summary_writer.add_audio(tag, audio, self.iteration)
