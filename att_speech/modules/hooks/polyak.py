from att_speech.modules.hooks.hook import TrainingLoopHook
from att_speech import utils
from copy import deepcopy


class PolyakDecay(TrainingLoopHook):
    def __init__(self, decay_rates, **kwargs):
        self.polyak_decay = decay_rates
        super(PolyakDecay, self).__init__(self, **kwargs)

    def pre_run(self, model, optimizer):
        st_dict = model.state_dict()
        good_polyak_names = []
        for decay in self.polyak_decay:
            polyak_dict_name = 'avg_state_dict_%f' % (decay)
            good_polyak_names.append(polyak_dict_name)
            if not hasattr(model, polyak_dict_name):
                setattr(model, polyak_dict_name, deepcopy(st_dict))
        for polyak_dict_name in list(model.__dict__.keys()):
            if (polyak_dict_name.startswith('avg_state_dict') and
                    polyak_dict_name not in good_polyak_names):
                print("Polyak deleting ", polyak_dict_name)
                delattr(model, polyak_dict_name)

    def post_optimizer_step(self, model, optimizer, current_iteration, loss):
        st = model.state_dict()
        for decay in self.polyak_decay:
            polyak_dict_name = 'avg_state_dict_%f' % (decay)
            polyak_dict = getattr(model, polyak_dict_name)
            for k in polyak_dict:
                polyak_dict[k] = (
                    decay*polyak_dict[k] +
                    (1 - decay)*st[k])

    def post_dev_eval(self, model, current_iteration, logger, save_dir,
                      dev_dataset):
        old_state = deepcopy(model.state_dict())

        for decay in self.polyak_decay:
            def progress_clb(*x):
                print(
                    'Polyak {}: Processing batch {}/{} ({} elements)'.format(
                        decay, *x))

            polyak_dict_name = 'avg_state_dict_%f' % (decay)
            polyak_dict = getattr(model, polyak_dict_name)
            print('Evaluating polyak %f decayed model' % (decay,))
            model.load_state_dict(polyak_dict)
            polyak_eval_result = utils.evaluate_greedy(
                dev_dataset, model, progress_callback=progress_clb)

            logger.make_step_log(
                '{}/dev_polyak_{}/'.format(save_dir, decay),
                current_iteration)
            for k, v in polyak_eval_result.iteritems():
                logger.log_scalar('_' + k, v)
            logger.end_log()

        model.load_state_dict(old_state)
