from att_speech import utils
from att_speech.config_utilities import ConfigParser, ConfigLinker


def get_val(dictionary, key, dict_name):
    if key not in dictionary:
        raise KeyError('%s has no %s key specified' % (dict_name, key))

    return dictionary[key]


class ConfigInstantiator(object):
    def __init__(self, objects_config, default_class_dict={},
                 default_modules_dict={}, name='', **kwargs):
        super(ConfigInstantiator, self).__init__(**kwargs)
        self.objects_config = objects_config
        self.default_class_dict = default_class_dict
        self.default_modules_dict = default_modules_dict
        self.cache = {}
        self.name = name

    def keys(self):
        return self.objects_config.keys()

    def _getitem(self, key, additional_parameters=None):
        if key not in self.cache:
            # make a copy since we may change the dict in the end
            opts = dict(get_val(self.objects_config, key, self.name))
            if 'class_name' not in opts:
                opts['class_name'] = self.default_class_dict[key]
            self.cache[key] = utils.contruct_from_kwargs(
                    opts, self.default_modules_dict.get(key),
                    additional_parameters)
        return self.cache[key]

    def __getitem__(self, key):
        return self._getitem(key)


class ConfigDict(ConfigInstantiator):
    def __init__(self, **kwargs):
        super(ConfigDict, self).__init__(kwargs)


class _ConstantDict(object):
    def __init__(self, v, **kwargs):
        super(_ConstantDict, self).__init__(**kwargs)
        self.v = v

    def __getitem__(self, k):
        return self.v

    def get(self, k, v=None):
        return self.v


class _Dict(dict):
    def __init__(self, **kwargs):
        super(_Dict, self).__init__(kwargs)


class Configuration(ConfigInstantiator):
    """
    Class responsible for instantiating object that are defined in config file.

    The class tries to be smart about the following modules:
    - Trainer will by default instantiate an 'att_speech.trainer.Trainer'
    - all items on the Data key will instantiate a 'att_speech.data.Data'
    - It will configure the Model key accorting to Dataset specification

    Args:
        config_path (str): Path pointing to the config file.
        modify_dict (dict): Optional dictionary
            representing config modifications.
        store_path (str): Optional path to store linked config.
    """

    default_class_dict = {
        'Trainer': 'Trainer',
        'Datasets': 'ConfigDict',
        'LMEvaluator': '_Dict',
    }
    default_modules_dict = {
        'Trainer': 'att_speech.trainer',
        'Datasets': 'att_speech.configuration',
        'Model': 'att_speech.models',
        'LMEvaluator': 'att_speech.configuration'
    }

    def __init__(self, config_path, modify_dict={}, store_path=None, **kwargs):
        config = ConfigParser(config_path).get_config(modify_dict)
        if store_path is not None:
            ConfigLinker(config).save_linked_config(store_path)
        super(Configuration, self).__init__(
            objects_config=config,
            default_class_dict=Configuration.default_class_dict,
            default_modules_dict=Configuration.default_modules_dict,
            name=config_path,
            **kwargs)

    def __getitem__(self, key):
        if key == 'Model':
            model_param = {
                'sample_batch': self['Datasets']['train'].sample_batch(),
                'num_classes': self['Datasets']['train'].num_classes(),
                'vocabulary': self['Datasets']['train'].dataset.vocabulary.itos
            }
            return self._getitem('Model', additional_parameters=model_param)
        else:
            return super(Configuration, self).__getitem__(key)


class Globals(object):
    """Global configuration objects."""
    cuda = True
