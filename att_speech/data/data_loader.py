from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data
from torch.autograd import Variable

import numpy as np

from att_speech import utils
from att_speech.fst_utils import batch_training_graph_matrices


def _combine_field(batch, indicies, field, lengths=None, dynamic_axis=0):
    if not lengths:
        lengths = [example[field].shape[dynamic_axis] for example in batch]
    lengths = np.array(lengths, 'int32')
    max_length = np.max(lengths)

    return (torch.from_numpy(np.stack(
            [np.lib.pad(
                batch[i][field],
                [(0, 0) if j != dynamic_axis else (0, max_length - lengths[i])
                 for j in range(batch[0][field].ndim)],
                'constant',
                constant_values=0)
             for i in indicies])),
            torch.from_numpy(lengths[indicies]))


def _kaldi_collate_fn(batch):
    feat_lengths = [example['feats'].shape[0] for example in batch]
    indicies = np.argsort(feat_lengths)[::-1]

    uttids = [batch[i]['uttid'] for i in indicies]
    spkids = [batch[i]['spkid'] for i in indicies]

    feats = _combine_field(batch, indicies, 'feats', dynamic_axis=0)
    texts = _combine_field(batch, indicies, 'text', dynamic_axis=0)

    if batch[0]['ivectors'] is not None:
        ivectors = _combine_field(
            batch, indicies, 'ivectors', dynamic_axis=0)[0]
    else:
        ivectors = None

    result = {'uttids': uttids,
              'texts': texts,
              'features': feats,
              'spkids': spkids,
              'ivectors': ivectors}

    if 'graph_matrices' in batch[0]:
        result['graph_matrices'] = batch_training_graph_matrices(
            [batch[i]['graph_matrices'] for i in indicies],
            nc_weight=-1e20, device='cpu')

    if batch and len(batch) > 0:
        for key in batch[0]:
            if key not in ['uttid', 'text', 'feats', 'spkid', 'ivectors',
                           'graph_matrices']:
                result[key + "s"] = [batch[i][key] for i in indicies]

    return result


class KaldiDatasetLoader(torch.utils.data.DataLoader):
    """
    KaldiDatasetLoader is a torch DataLoader equipped with collate_fn
    designed for this dataset
    """

    # pylint: disable=too-few-public-methods
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = _kaldi_collate_fn
        super(KaldiDatasetLoader, self).__init__(*args, **kwargs)


class ConfigurableData(KaldiDatasetLoader):
    def __init__(self, dataset, **kwargs):
        if 'transforms' in dataset:
            dataset['transforms'] = [utils.contruct_from_kwargs(t)
                                     for t in dataset['transforms']]
        dataset = utils.contruct_from_kwargs(dataset)
        super(ConfigurableData, self).__init__(dataset=dataset, **kwargs)

    def sample_batch(self):
        data = next(iter(self))
        result = {'features': Variable(data['features'][0]),
                  'features_lengths': Variable(data['features'][1]),
                  'spkids': data['spkids'],
                  'ivectors': (
                    Variable(data['ivectors']) if data['ivectors'] is not None
                    else None)}
        for key in data:
            if key not in ['features', 'spkids', 'features_lengths', 'uttids',
                           'ivectors']:
                result[key] = data[key]

        return result

    def num_classes(self):
        return self.dataset.num_classes()

    def speakers(self):
        return set(self.dataset.uttid_to_spkid.values())
