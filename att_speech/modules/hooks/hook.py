class TrainingLoopHook(object):
    def __init__(self, priority=None):
        if priority:
            self.priority = priority
        else:
            self.priority = 0

    def pre_run(self, model, optimizer):
        pass

    def pre_train_forward(self, model, optimizer, current_iteration):
        pass

    def pre_backward(self, model, optimizer, current_iteration, loss):
        pass

    def post_backward(self, model, optimizer, current_iteration, loss):
        pass

    def post_optimizer_step(self, model, optimizer, current_iteration, loss):
        pass

    def post_dev_eval(self, model, current_iteration, logger, save_dir,
                      dev_dataset):
        pass
