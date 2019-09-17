import glob
import re
import os


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def last_yaml(path):
    """The name of the recent train_configX.yaml"""
    cfg = [f for f in os.listdir(path) if re.match(r'train_config\d+.yaml$', f)]
    return natural_sort(cfg)[-1]


name_map = {
    'mono_fst_ctc':             'ctc',
    'bi_fst':                   'ctcg_bi_cde',
    'bi_fst_lut':               'ctcg_bi',
    'bi_fst_ctc':               'ctc_bi_cde',
    'bi_fst_ctc_lut':           'ctc_bi',
    'bi_fst_contextblanks':     'ctcgb_bi_cde',
    'bi_fst_lut_contextblanks': 'ctcgb_bi',
    'tri_fst_ctc':              'ctc_tri_cde',
    'tri_fst_ctc_lut':          'ctc_tri',
}

runs_mono_bi = '/pio/scratch/1/alan/att_speech/runs_fst'
runs_tri = '/pio/scratch/1/alan/att_speech/runs_tri'

for name, yaml in name_map.items():
    if 'tri' in name:
        path = '%s/%s' % (runs_tri, name)
    else:
        path = '%s/%s_R3' % (runs_mono_bi, name)
    old_yaml = last_yaml(path)
    os.system('cp %s/%s %s.yaml' % (path, last_yaml(path), yaml))
