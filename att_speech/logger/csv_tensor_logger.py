from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from datetime import datetime
import sys
import time

from . import TensorLogger


class CsvSummaryWriter(object):
    def __init__(self, path):
        self.path = path
        # We will populate scalars names list during first log step so we want
        # to treat it in a different way.
        self.initialized = False
        self.scalars_names = ["iteration", "begin_timestp", "end_timestp"]
        self.iteration = -1
        # Mapping scalar_name to scalar_value, to be written to csv
        self.scalars = {}

    def add_scalar(self, name, value, iteration):
        if self.initialized:
            self._add_scalar_initialized(name, value, iteration)
        else:
            self._add_scalar_not_initialized(name, value, iteration)

    def flush(self):
        # Creating columns names line in the file
        if not self.initialized:
            self.out_file = open(
                self.path + "logs" + str(datetime.now()).replace(" ", "_"),
                "w")
            self._write_scalars_names()
        self.initialized = True
        self.scalars["end_timestp"] = time.mktime(datetime.now().timetuple())
        self._write_scalars()
        self.scalars = {}

    def _write_scalars_names(self):
        """
        Write new, first line to self.out_file with scalars name from
        self.scalars_names
        """
        self._write_scalars_universal(lambda x: x)

    def _write_scalars(self):
        """ Write new line to self.out_file with scalars from self.scalars """
        self._write_scalars_universal(
            lambda name: (float(self.scalars[name]) if name in self.scalars
                          else None))

    def _add_scalar_not_initialized(self, name, value, iteration):
        if name not in self.scalars_names:
            self.scalars_names.append(name)
        self._add_scalar_initialized(name, value, iteration)

    def _add_scalar_initialized(self, name, value, iteration):
        if name not in self.scalars_names:
            print("Name %s not recognized to log." % name, file=sys.stderr)
            return
        if iteration < self.iteration:
            raise RuntimeError(
                "iteration argument for logger should be non decreasing, got "
                "%d, was %d before" % (iteration, self.iteration))
        elif iteration > self.iteration:
            # Asserting that self.scalars is empty, making sure flush was
            # called
            assert not self.scalars
            self.iteration = iteration
            self.scalars["iteration"] = iteration
            self.scalars["begin_timestp"] = time.mktime(
                datetime.now().timetuple())
        self.scalars[name] = value

    # Writing first line with columns names and the following ones are
    # basically the same exept for the fact that in the former we want to write
    # the names themselves while in the latter we write corresponding scalars.
    # To avoid copying the code both cases are wrapped in one function and the
    # difference between them is name2scalar function.
    def _write_scalars_universal(self, name2scalar):
        # Writing first column, without comma
        name = self.scalars_names[0]
        scalar = name2scalar(name)
        if scalar is None:
            print("Scalar %s not found while logging step %d, writing NaN" % (
                name, iteration))
            self.out_file.write("NaN")
        else:
            self.out_file.write(str(scalar))
        # Writing following columns, with comme before each
        for name in self.scalars_names[1:]:
            self.out_file.write(",")
            scalar = name2scalar(name)
            if scalar is None:
                print("Scalar %s not found while logging step %d, writing NaN"
                    % (name, self.iteration))
                self.out_file.write("NaN")
            else:
                self.out_file.write(str(scalar))
        self.out_file.write("\n")
        self.out_file.flush()


class CsvTensorLogger(TensorLogger):
    # global cache of sumary writetrs
    summary_writers = {}
    
    def get_summary_writer(self, path):
        if path not in self.summary_writers:
            self.summary_writers[path] = CsvSummaryWriter(path)
        return self.summary_writers[path]

    def end_log(self):
        self.ensure_during_log()
        if self.log_state == self.STEP_LOG_STATE:
            # print(type(self))
            # print(type(self.summary_writer))
            self.summary_writer.flush()
            self.summary_writer = None
            self.iteration = None
        self.log_state = self.NO_LOG_STATE


    def log_histogram(self, name, values):
        raise NotImplementedError(
            "log_histogram is not implemented for CsvTensorLogger")

    def log_image(self, tag, img):
        raise NotImplementedError(
            "log_image is not implemented for CsvTensorLogger")

    def log_audio(self, tag, audio):
        raise NotImplementedError(
            "log_audio is not implemented for CsvTensorLogger")
