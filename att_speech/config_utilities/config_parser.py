from __future__ import absolute_import
import yaml
import os.path
import copy


class ConfigParser:
    PARENT_NODE = 'parent'

    def __init__(self, root_config_file):
        self.root_config_file = root_config_file
        self.root_config_dict = None

    def apply_changes_in_config(self, config_dict, changes_dict):
        for key, value in changes_dict.items():
            if isinstance(value, dict) and isinstance(config_dict[key], dict):
                self.apply_changes_in_config(config_dict[key], value)
            else:
                config_dict[key] = value

    def read_config(self, config_file):
        with open(config_file) as config_stream:
            config_dict = yaml.load(config_stream)
            if self.PARENT_NODE in config_dict:
                parent_file = os.path.expandvars(config_dict[self.PARENT_NODE])
                changes_dict = config_dict
                config_dict = self.read_config(parent_file)
                self.apply_changes_in_config(config_dict, changes_dict)
            return config_dict

    def init_config_dict(self):
        if self.root_config_dict is None:
            self.root_config_dict = self.read_config(self.root_config_file)

    def modify_config_node(self, config_dict, path, value):
        path_parts = path.split('.')
        path_node = config_dict
        for path_part in path_parts[:-1]:
            path_node = path_node[path_part]
        path_node[path_parts[-1]] = yaml.load(value)

    def get_config(self, modify_dict={}):
        self.init_config_dict()
        config_dict = copy.deepcopy(self.root_config_dict)
        for path, value in modify_dict.iteritems():
            self.modify_config_node(config_dict, path, value)
        return config_dict
