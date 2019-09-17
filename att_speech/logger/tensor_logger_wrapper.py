from __future__ import absolute_import

from . import TensorLogger, CsvTensorLogger


class TensorLoggerWrapper(object):
    def __init__(self):
        self.tensor_logger = TensorLogger()
        self.csv_tensor_logger = CsvTensorLogger()

    def is_currently_logging(self):
        return (self.tensor_logger.is_currently_logging and
                self.csv_tensor_logger.is_currently_logging())

    def make_step_log(self, log_dir, iteration):
        self.tensor_logger.make_step_log(log_dir, iteration)
        self.csv_tensor_logger.make_step_log(log_dir, iteration)

    def make_null_log(self):
        self.tensor_logger.make_null_log()
        self.csv_tensor_logger.make_null_log()

    def end_log(self):
        self.tensor_logger.end_log()
        self.csv_tensor_logger.end_log()

    def log_scalar(self, name, value):
        self.tensor_logger.log_scalar(name, value)
        self.csv_tensor_logger.log_scalar(name, value)

    def log_histogram(self, name, values):
        self.tensor_logger.log_histogram(name, values)
        self.csv_tensor_logger.log_histogram(name, values)

    def log_image(self, tag, img):
        self.tensor_logger.log_image(tag, img)
        self.csv_tensor_logger.log_image(tag, img)

    def log_audio(self, tag, audio):
        self.tensor_logger.log_audio(tag, audio)
        self.csv_tensor_logger.log_audio(tag, audio)

    def __getattr__(self, name):
        if name == "iteration":
            return self.tensor_logger.iteration
        else:
            return getattr(self, name)
