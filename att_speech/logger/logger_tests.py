from __future__ import absolute_import

import unittest
from shutil import rmtree
from os import listdir

from . import DefaultTensorLogger


class LoggerTests(unittest.TestCase):
    def tearDown(self):
        rmtree('logger_tests')

    def test_singleton(self):
        logger1 = DefaultTensorLogger()
        logger2 = DefaultTensorLogger()
        logger1.make_step_log('logger_tests/test1', 1)
        logger1.end_log()
        self.assertEqual(logger2.get_current_iteration('logger_tests/test1'),
                         1)

    def test_iteration(self):
        logger = DefaultTensorLogger()
        logger.make_step_log('logger_tests/test2', 2)
        logger.end_log()
        for i in range(11):
            logger.make_null_log()
            logger.end_log()
        logger.make_step_log('logger_tests/test2', 3)
        logger.end_log()
        self.assertEqual(logger.get_current_iteration('logger_tests/test2'),
                         3)

    def test_logging(self):
        logger = DefaultTensorLogger()
        for i in range(3):
            logger.make_step_log('logger_tests/test3', i)
            logger.log_scalar('x', i + 1)
            logger.log_scalar('y', i + 3)
            logger.end_log()
        self.assertTrue(len(listdir('logger_tests/test3')) > 0)


if __name__ == '__main__':
    unittest.main()
