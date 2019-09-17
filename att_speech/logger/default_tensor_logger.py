from __future__ import absolute_import

from . import TensorLoggerWrapper


class DefaultTensorLogger(object):
    log_instance = None

    def __init__(self):
        if DefaultTensorLogger.log_instance is None:
            DefaultTensorLogger.log_instance = TensorLoggerWrapper()

    def __getattr__(self, name):
        return getattr(self.log_instance, name)
