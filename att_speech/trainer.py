from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from copy import deepcopy
from collections import OrderedDict

from datetime import datetime
import time
import numpy as np

from torch.autograd import Variable

from torch.utils.data.sampler import BatchSampler

from att_speech import utils
from att_speech.configuration import Globals
from att_speech.checkpointer import Checkpointer
from att_speech.logger import DefaultTensorLogger, TensorLogger

from att_speech.data import SortByLengthSampler

DEFAULT_LR_SCHEDULER = {
        'class_name': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'factor': 0.5,
        'patience': 3
        }


class Trainer(object):
    def __init__(self, num_epochs, learning_rate,
                 optimizer_name, optimizer_kwargs={},
                 hooks=None,
                 learning_rate_scheduler=DEFAULT_LR_SCHEDULER,
                 checkpointer=None,
                 dump_logits_on_eval=False,
                 log_frequency=100,
                 init_phase_iters=0,
                 log_layers_gradient=False,
                 print_num_decoding_samples=0,
                 additional_log_filters=[], # list of strings, additional names
                 **kwargs):
        super(Trainer, self).__init__(**kwargs)

        if checkpointer is None:
            checkpointer = {}
        if learning_rate_scheduler is None:
            learning_rate_scheduler = DEFAULT_LR_SCHEDULER

        self.dump_logits_on_eval = dump_logits_on_eval
        self.log_frequency = log_frequency
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.learning_rate_scheduler_params = learning_rate_scheduler
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.init_phase_iters = init_phase_iters
        self.best_values = {}
        self.current_iteration = 0
        self.gradient_logger = TensorLogger()
        self.log_layers_gradient = log_layers_gradient
        self.print_num_decoding_samples = print_num_decoding_samples
        self.checkpointer = Checkpointer(**checkpointer)
        self.additional_log_filters = additional_log_filters

        self.evaluation_function = utils.evaluate_greedy
        self.evaluation_function_in_train_mode = \
            utils.evaluate_greedy_in_train_mode
        self.eval_metric_name = 'CER'

        self.hooks = []

        if hooks:
            self.hooks = sorted(
                [utils.contruct_from_kwargs(
                    {'class_name': hname},
                    'att_speech.modules.hooks',
                    hparams
                    )
                for hname, hparams in hooks.iteritems()],
                key=lambda h: h.priority)

    def _log_train_grads(self, logger, grad_stats, grad_logger, save_dir):
        for stat in ('max', 'min', 'mean'):
            grad_logger.make_step_log('{}/norm_{}/'.format(save_dir, stat),
                                      logger.iteration)
            if self.log_layers_gradient:
                for p, v in grad_stats.get(stat).iteritems():
                    grad_logger.log_scalar('norm_{} {}'.format(stat, p), v)
            aggr = grad_stats.get(stat, aggregate_by=stat)
            grad_logger.log_scalar('_gradients', aggr)
            grad_logger.end_log()

        filter_funs = {'conv': lambda p: '.conv.' in p and 'batch_norm' not in p,
                       'rnn': lambda p: '.rnn.' in p and 'batch_norm' not in p,
                       'fc': lambda p: '.fc.' in p and 'batch_norm' not in p,
                       'bn': lambda p: '.batch_norm.' in p}
        for name in self.additional_log_filters:
            filter_funs[name] = (
                lambda p: '.' + name + '.' in p and 'batch_norm' not in p)
        for name, fun in filter_funs.items():
            for stat in ('max', 'min', 'mean'):
                grad_logger.make_step_log('{}/norm_{}/'.format(save_dir, stat),
                                          logger.iteration)
                aggr = grad_stats.get(stat, aggregate_by=stat,
                                      param_filter_fun=fun)
                grad_logger.log_scalar('_gradients_{}'.format(name), aggr)
                grad_logger.end_log()

    def _log_train_batch(self, logger, model, loss, optimizer, save_dir):
        if isinstance(loss, dict):
            for key in loss:
                if loss[key].numel() == 1:
                    logger.log_scalar('_'+key, loss[key].item())
        else:
            logger.log_scalar('_loss', loss.item())

        for i, param_group in enumerate(optimizer.param_groups):
            logger.log_scalar('_learning_rate{}'.format(i), param_group['lr'])

    def _log_dev_eval(self, logger, evaluated):
        for key in evaluated:
            logger.log_scalar('_' + key, evaluated[key])

    def run(self, save_dir, model, train_dataset, eval_datasets=None,
            saved_state=None, debug_skip_training=False, decode_first=False, ):
        if saved_state:
            model.load_state(saved_state['state_dict'])
            for k in saved_state:
                if k.startswith('avg_state_dict'):
                    print("Loading poyak's ", k)
                    setattr(model, k, saved_state[k])
        if eval_datasets is None:
            eval_datasets = {}
        if Globals.cuda:
            model.cuda()
        optimizer = getattr(
            torch.optim,
            self.optimizer_name)(model.get_parameters_for_optimizer(),
                                 lr=self.learning_rate,
                                 **self.optimizer_kwargs)
        if saved_state:
            optimizer.load_state_dict(saved_state['optimizer'])

        self.learning_rate_scheduler_params['optimizer'] = optimizer
        learning_rate_scheduler = utils.contruct_from_kwargs(
            self.learning_rate_scheduler_params)
        if saved_state:
            self.current_iteration = saved_state['current_iteration']
            start_epoch = saved_state['epoch'] + 1
            learning_rate_scheduler.load_state_dict(
                saved_state['learning_rate_scheduler'])
        else:
            self.current_iteration = 0
            start_epoch = 0

        self.checkpointer.set_save_dir(save_dir)

        for h in self.hooks:
            h.pre_run(model=model, optimizer=optimizer)

        if self.init_phase_iters and start_epoch < self.init_phase_iters:
            sorted_train_dataset = deepcopy(train_dataset)
            sampler = BatchSampler(SortByLengthSampler(train_dataset.dataset),
                                   train_dataset.batch_size, False)
            sorted_train_dataset.batch_sampler = sampler

            for i in range(start_epoch, self.init_phase_iters):
                self.iterate_epoch(
                    i, save_dir, model, sorted_train_dataset, eval_datasets,
                    optimizer, learning_rate_scheduler)
            start_epoch = self.init_phase_iters

        for epoch in range(start_epoch, self.num_epochs):
            self.iterate_epoch(
                epoch, save_dir, model, train_dataset, eval_datasets,
                optimizer, learning_rate_scheduler,
                decode_only=decode_first,
                debug_skip_training=debug_skip_training)
            decode_first = False

    def iterate_epoch(self, epoch, save_dir, model, train_dataset,
                      eval_datasets, optimizer, learning_rate_scheduler,
                      decode_only=False, debug_skip_training=False):
        startTime = datetime.now()
        logger = DefaultTensorLogger()
        batch_processed = 0
        data_len = len(train_dataset)
        model.train()
        gradient_stats = utils.GradientStatsCollector()
        bend = time.time()

        for batch in train_dataset:
            if decode_only:
                break
            bstart = time.time()
            data_load_time = bstart - bend
            if (self.current_iteration % self.log_frequency) == 0:
                logger.make_step_log('{}/train/'.format(save_dir),
                                     self.current_iteration)
            else:
                logger.make_null_log()

            self.current_iteration += 1

            feature_lens = Variable(batch['features'][1])
            features = Variable(batch['features'][0])
            if Globals.cuda:
                features = features.cuda()
            ivectors = batch['ivectors']
            if ivectors is not None:
                ivectors = Variable(ivectors)
                if Globals.cuda:
                    ivectors = ivectors.cuda()

            text_lens = Variable(batch['texts'][1])
            texts = Variable(batch['texts'][0])
            spkids = batch['spkids']

            args = {}

            for key in batch:
                if key not in ['spkids', 'texts', 'features', 'uttids',
                               'ivectors']:
                    args[key] = batch[key]

            for h in self.hooks:
                h.pre_train_forward(
                    model=model,
                    optimizer=optimizer,
                    current_iteration=self.current_iteration)

            loss = model.forward(
                features, feature_lens, spkids, texts, text_lens, ivectors,
                **args)

            loss_dict = loss

            should_skip_opt_step = False
            for h in self.hooks:
                should_skip_opt_step = (
                    should_skip_opt_step or
                    h.pre_backward(
                        model=model,
                        current_iteration=self.current_iteration,
                        optimizer=optimizer,
                        loss=loss_dict['loss']))

            optimizer.zero_grad()
            loss_dict['loss'].backward()

            gradient_stats.add(model)
            if logger.is_currently_logging():
                self._log_train_grads(logger, gradient_stats,
                                      self.gradient_logger, save_dir)
                gradient_stats = utils.GradientStatsCollector()

            for h in self.hooks:
                should_skip_opt_step = (
                    should_skip_opt_step or
                    h.post_backward(
                        model=model,
                        current_iteration=self.current_iteration,
                        optimizer=optimizer,
                        loss=loss_dict['loss']))

            if should_skip_opt_step:
                print(
                    'Ep. {: >3} | Batch {: >5}/{} | Loss {: >8.2f} | '
                    'Skipping optimizer step due to hooks'
                    ''.format(epoch, batch_processed, data_len,
                              loss_dict["loss"].item(),))
            else:
                optimizer.step()

            for h in self.hooks:
                h.post_optimizer_step(
                    model=model,
                    current_iteration=self.current_iteration,
                    optimizer=optimizer,
                    loss=loss_dict['loss'])

            if logger.is_currently_logging():
                self._log_train_batch(logger, model, loss_dict, optimizer,
                                      save_dir)
            logger.end_log()
            batch_processed += 1
            bend = time.time()
            btime = bend - bstart

            print('Ep. {: >3} | Batch {: >5}/{} | Loss {: >8.2f} | '
                  'Time {: >.2f}s | DataLoad {: >.5f}s | '
                  '{: >.1f} tokens/s | Lr {: >7.5f} | Elapsed {}'.format(
                    epoch, batch_processed, data_len, loss_dict["loss"].item(),
                    btime, data_load_time, feature_lens.sum().item() / btime,
                    optimizer.param_groups[0]['lr'],
                    str(datetime.now() - startTime).split('.')[0]))

            if debug_skip_training:
                break

        print("Saving last epoch")
        self.checkpointer.checkpoint(self.current_iteration, epoch,
                                     model, optimizer, learning_rate_scheduler)

        print('Epoch {} ended. Evaluating on dev dataset'.format(epoch))

        def progress_clb(*x):
            print('Processing batch {}/{} ({} elements)'.format(*x))

        def get_logits_dumper(name):
            if self.dump_logits_on_eval:
                return utils.LogitsDumper(
                        '{}/logits_{}'.format(save_dir, name),
                        self.current_iteration)
            else:
                return None

        eval_result = self.evaluation_function(
            eval_datasets['dev'], model, progress_callback=progress_clb,
            logits_dumper=get_logits_dumper('dev'),
            print_num_samples=self.print_num_decoding_samples)

        logger.make_step_log('{}/dev/'.format(save_dir),
                             self.current_iteration)
        self._log_dev_eval(logger, eval_result)
        logger.end_log()

        for h in self.hooks:
            h.post_dev_eval(
                model=model,
                current_iteration=self.current_iteration,
                logger=logger,
                save_dir=save_dir,
                dev_dataset=eval_datasets['dev'])

        for key in eval_result:
            print('Epoch {} {}: {}'.format(epoch, key, eval_result[key]))
            self.checkpointer.try_checkpoint_best(
                    key.replace('_', '-'), eval_result[key],
                    self.current_iteration, epoch,
                    model, optimizer, learning_rate_scheduler)

        learning_rate_scheduler.step(
            metrics=eval_result[self.eval_metric_name], epoch=epoch)
        elapsed = datetime.now() - startTime
        print("Current epoch took {}".format(elapsed))
