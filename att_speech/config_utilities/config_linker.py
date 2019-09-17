from __future__ import absolute_import
import yaml
import os
from att_speech import utils


class ConfigLinker:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def save_linked_config(self, output_file):
        dir_name = os.path.dirname(output_file)
        if len(dir_name) > 0:
            utils.ensure_dir(dir_name)
        with open(output_file, 'w') as file_stream:
            yaml.dump(self.config_dict, file_stream, default_flow_style=False)
