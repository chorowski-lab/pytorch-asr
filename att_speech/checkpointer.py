from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from att_speech import utils


class Checkpointer(object):
    '''
    The checkpointer. Stores:
        * last_n checkpoints (with filenames checkpoint_STEPNUM.pkl)
        * one checkpoint every n_hours (with filenames: checkpoint_STEPNUM.pkl)
        * one best checkpoint for every logged channel
          (filename: best_STEPNUM_CHANNELNAME_VALUE.pkl)
    '''
    def __init__(self, last_n=3, every_n_hours=8):
        self.last_n = last_n
        self.every_n_seconds = int(60 * 60 * every_n_hours)
        self.last_reload_iteration = None
        self.checkpoints = []
        self.best_checkpoints = {}

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir

    def load_checkpointer_state(self, iteration):
        if iteration == self.last_reload_iteration:
            return
        filenames = [f[:-4] for f in os.listdir(self.create_path())]
        self.checkpoints = (
            [(f, int(os.path.getmtime(self.create_path(f))))
             for f in filenames if f.startswith('checkpoint_') and
             f.endswith('.pkl')])
        self.checkpoints = sorted(self.checkpoints,
                                  key=lambda x: int(x[0][x[0].rindex('_')+1:]))

        self.best_checkpoints = {}
        best_checkpoints = [f for f in filenames if f.startswith('best_')]
        for f in best_checkpoints:
            fields = f.split('_')
            cerval = float(fields[3])
            chann = fields[2]
            self.best_checkpoints[chann] = (cerval, f)
        self.last_reload_iteration = iteration

    def remove_unnecessary_checkpoints(self):
        if not self.checkpoints:
            return
        last_safe = self.checkpoints[0]
        removeable_checkpoints = []
        for checkpoint in self.checkpoints[1:]:
            if checkpoint[1] - last_safe[1] >= self.every_n_seconds:
                last_safe = checkpoint
            else:
                removeable_checkpoints += [checkpoint]
        for rem in removeable_checkpoints[:(-1*self.last_n)]:
            self.remove(rem[0])

    def create_path(self, filename=None):
        assert self.save_dir, "Checkpointer: Save dir not set"
        if filename is None:
            return os.path.join(self.save_dir, 'checkpoints')
        else:
            return os.path.join(self.save_dir,
                                'checkpoints', '{}.pkl'.format(filename))

    def save(self, filename, current_iteration, epoch, model, optimizer,
             scheduler):
        print("Saving {}".format(filename))

        possible_filename = 'checkpoint_{}'.format(current_iteration)
        oname = self.create_path(filename)
        if any((k[0] == possible_filename for k in self.checkpoints)):
            print("Iteration {} already saved, making hard link"
                  .format(current_iteration))
            os.link(self.create_path(possible_filename), oname)
            return
        else:
            temp_path = self.create_path('.{}.pkl.temporary'.format(filename))
            state_dict = {
                'current_iteration': current_iteration,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate_scheduler': scheduler.state_dict(),
            }
            for k in model.__dict__:
                if k.startswith('avg_state_dict'):
                    print("Adding polyak", k)
                    state_dict[k] = getattr(model, k)
            utils.ensure_dir(os.path.dirname(oname))
            torch.save(state_dict, temp_path)
            os.rename(temp_path, oname)

    def remove(self, filename):
        print("Removing {}".format(filename))
        oname = self.create_path(filename)
        os.remove(oname)

    def try_checkpoint_best(self, name, value,
                            iteration_num, epoch_num,
                            model, optimizer, scheduler):
        self.load_checkpointer_state(iteration_num)
        if name not in self.best_checkpoints \
           or value < self.best_checkpoints[name][0]:
            point_name = 'best_{}_{}_{}'.format(iteration_num, name, value)
            self.save(point_name,
                      iteration_num, epoch_num,
                      model, optimizer, scheduler)
            if name in self.best_checkpoints:
                self.remove(self.best_checkpoints[name][1])
            self.best_checkpoints[name] = (value, point_name)

    def checkpoint(self, interation_num, epoch_num,
                   model, optimizer, scheduler):
        point_name = 'checkpoint_{}'.format(interation_num)
        self.save(point_name,
                  interation_num, epoch_num,
                  model, optimizer, scheduler)
        self.load_checkpointer_state(interation_num)
        self.remove_unnecessary_checkpoints()
