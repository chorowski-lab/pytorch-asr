from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import importlib
import os
import re
from collections import defaultdict, OrderedDict

import kaldi_io
import numpy as np
import torch
from torch.autograd import Variable


def edit_distance(x, y):
    """Returns the edit distance between sequences x and y. We are using
    dynamic programming to compute the minimal number of operations needed
    to transform sequence x into y."""
    dp = np.zeros((len(x)+1, len(y)+1), dtype='int64')
    for i in range(len(x)+1):
        dp[i][0] = i
    for i in range(len(y)+1):
        dp[0][i] = i
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,  # insertion
                dp[i][j-1] + 1,  # deletion
                dp[i-1][j-1] + (0 if x[i-1] == y[j-1] else 1),
            )
    return dp[-1][-1]

def edit_distance_with_stats(x, y):
    dp = np.zeros((len(x)+1, len(y)+1), dtype='int64')
    op = np.zeros((len(x)+1, len(y)+1), dtype='int64')
    for i in range(len(x)+1):
        dp[i][0] = i
        op[i][0] = 0
    for i in range(len(y)+1):
        dp[0][i] = i
        op[0][i] = 1
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            operations = (
                dp[i-1][j] + 1,  # insertion
                dp[i][j-1] + 1,  # deletion
                dp[i-1][j-1] + (0 if x[i-1] == y[j-1] else 1),
            )
            choosen_op = np.argmin(operations)
            op[i][j] = choosen_op
            dp[i][j] = operations[choosen_op]
    i = len(x)
    j = len(y)
    operations = [0, 0, 0]
    while i >= 0 and j >= 0:
        old_op = op[i][j]
        ni = i if old_op == 1 else i - 1
        nj = j if old_op == 0 else j - 1
        if dp[i][j] > dp[ni][nj]:
            operations[old_op] += 1
        i = ni
        j = nj
    return dp[-1][-1], {'ins': operations[0], 'del': operations[1], 'sub': operations[2]}


def word_error_rate(x, y):
    """Returns the word error rate between sequences x and y."""
    return 1.0 * edit_distance(x, y) / len(x)


def get_class(str_or_class, default_mod=None):
    if isinstance(str_or_class, (str, unicode)):
        parts = str_or_class.split('.')
        mod_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        if mod_name:
            mod = importlib.import_module(mod_name)
        elif default_mod is not None:
            mod = importlib.import_module(default_mod)
        else:
            raise ValueError('Specify a module for %s' % (str_or_class,))
        return getattr(mod, class_name)
    else:
        return str_or_class


def contruct_from_kwargs(object_kwargs, default_mod=None,
                         additional_parameters=None):
    object_kwargs = dict(object_kwargs)
    class_name = object_kwargs.pop('class_name')
    klass = get_class(class_name, default_mod)
    if additional_parameters:
        object_kwargs.update(additional_parameters)
    obj = klass(**object_kwargs)
    return obj


def uniq(inlist):
    """
    Behaves like UNIX uniq command - removes repeating items.
    Returns list of (start, end) pairs such that list[start:end] has only
    one distinct element
    """
    if inlist == []:
        return []

    outl = []
    current_start = 0
    current_element = inlist[0]
    for i, elem in enumerate(inlist[1:], start=1):
        if current_element != elem:
            outl.append((current_start, i))
            current_start = i
            current_element = elem
    outl.append((current_start, i+1))
    return outl


class RunningStatistics(object):
    '''
    Numericably stable, one-pass computation of mean and variance. Based on
    https://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self):
        self._m = 0
        self._s = 0
        self._k = 0

        super(RunningStatistics, self).__init__()

    def add(self, data):
        '''
        Add data to running statistics
        :param data: numpy array
        '''
        data = np.ravel(data)
        if self._k == 0:
            self._m = data[0]
            self._s = 0
            self._k = 1
            data = data[1:]
        for value in data:
            self._k += 1
            difference = float(value - self._m)

            self._m = self._m + difference/self._k
            self._s = self._s + difference*(value - self._m)

    def mean(self):
        '''
        Returns current mean
        '''
        return self._m

    def variance(self):
        '''
        Returns current variance
        '''
        return self._s / (self._k - 1) if self._k > 1 else 0.0


class GradientStatsCollector(object):
    def __init__(self):
        self.gradients = OrderedDict()
        super(GradientStatsCollector, self).__init__()

    def add(self, model):
        for p, v in model.named_parameters():
            if v.grad is not None:
                self._add_grad(p, v.grad)

    def _add_grad(self, name, value):
        _norm = value.norm().item()
        if name in self.gradients:
            self.gradients[name] += [_norm]
        else:
            self.gradients[name] = [_norm]

    def get(self, stat='max', aggregate_by=None, param_filter_fun=None):
        '''
        Get a particular statistic

        Args:
            stat: str, how to aggregate over time (max, min, mean, median, var)
            aggregate_by: str, how to aggregate over params (max, min, etc.)
            param_filter_fun: lambda, aggregate params filtered by their name
        '''
        stats = [(p, getattr(np, stat)(v)) for (p, v) in self.gradients.iteritems()]
        if param_filter_fun:
            stats = [(p,v) for (p,v) in stats if param_filter_fun(p)]
        if aggregate_by:
            if stats:
                return getattr(np, aggregate_by)(zip(*stats)[1])
            else:
                return 0
        else:
            return OrderedDict(stats)


class LogitsDumper(object):
    def __init__(self, path, num_iter):
        ensure_dir(path)
        self.filename = os.path.join(path, '{}.ark.temp'.format(num_iter))
        self.end_filename = os.path.join(path, '{}.ark'.format(num_iter))

    def start(self):
        self.owriter = kaldi_io.BaseFloatMatrixWriter('ark:'+self.filename)

    def add_batch(self, uttids, logits):
        logits = logits.data.cpu().numpy()
        for i in np.argsort(uttids):
            self.owriter[uttids[i]] = logits[:, i, :]

    def end(self):
        del self.owriter
        os.rename(self.filename, self.end_filename)


def do_evaluate(dataset, model, output_callback=None,
                progress_callback=None, model_in_eval=True,
                generate_data_losses=False, logits_dumper=None,
                print_num_samples=0):
    dataset_length = len(dataset)

    if model_in_eval:
        model.eval()
    else:
        model.train()
    is_cuda = next(model.parameters()).is_cuda

    wer_sum = 0.
    wer_len = 0.
    cer_sum = 0.
    cer_len = 0.
    len_ratio = 0.
    sample_cnt = 0
    # losses_stats = RunningStatistics()
    losses_stats = defaultdict(lambda: RunningStatistics())

    if logits_dumper:
        logits_dumper.start()
    for j, batch in enumerate(dataset):
        feature_lens = Variable(batch['features'][1])
        features = Variable(batch['features'][0])

        text_lens = Variable(batch['texts'][1])
        texts = Variable(batch['texts'][0])

        speakers = batch['spkids']

        ivectors = batch['ivectors']
        if ivectors is not None:
            ivectors = Variable(ivectors)
            if is_cuda:
                ivectors = ivectors.cuda()

        args = {}
        for key in batch:
            if key not in ['features', 'texts', 'spkids', 'uttids',
                           'ivectors']:
                args[key] = batch[key]

        if is_cuda:
            features = features.cuda()

        if progress_callback:
            progress_callback(j, dataset_length, features.size()[0])
        if generate_data_losses:
            decoded = model.decode(
                features, feature_lens, speakers, texts, text_lens, {},
                {'return_texts_and_generated_loss': True,
                 'return_logits_text_diff': True}, ivectors=ivectors,
                **args)
            out, loss, text_loss, generated_loss, logits_diff = (
                decoded['decoded'], decoded['loss'], decoded['text_loss'],
                decoded['generated_loss'], decoded['logits_text_diff'])
        else:
            decoded = model.decode(
                features, feature_lens, speakers, texts, text_lens,
                ivectors=ivectors,
                **args)
            out = decoded['decoded']
            loss = decoded['loss']
            # loss = model.forward(features, feature_lens, speakers,
            #                      texts, text_lens, **args)
        if logits_dumper:
            logits_dumper.add_batch(batch['uttids'], decoded['logits'])

        if isinstance(loss, dict):
            for key in loss:
                losses_stats[key].add(loss[key].item())
        else:
            losses_stats['loss'].add(loss.item())

        if j == 0 and 'decoded_frames' in decoded:
            def clean(c):
                return chr(176) if c == '<pad>' else c
            decoded_frames = decoded['decoded_frames'][:print_num_samples]
            for i, elem in enumerate(decoded_frames):
                sample_chars, _, _ = (
                    dataset.dataset.ids_to_chars_words_sentence(elem))
                _, _, ref_sentence = (
                    dataset.dataset.ids_to_chars_words_sentence(
                        texts[i][:text_lens[i]]))
                sample_chars = [clean(c) for c in sample_chars]
                print('Ref:    ', ref_sentence)
                print('Decode: ', ''.join(sample_chars))
                if i + 1 == print_num_samples:
                    break

        for i, elem in enumerate(out):
            texts = batch['texts'][0]
            text_lens = batch['texts'][1]
            uttid = batch['uttids'][i]
            (decoded_chars, decoded_words, decoded_sentence
             ) = dataset.dataset.ids_to_chars_words_sentence(
                 elem, ignore_noise=True)

            (ref_chars, ref_words, ref_sentence
             ) = dataset.dataset.ids_to_chars_words_sentence(
                 texts[i][:text_lens[i]], ignore_noise=True)
            _wer_sum, _wer_stat = edit_distance_with_stats(decoded_words,
                                     ref_words)
            _wer_len = len(ref_words)
            _cer_sum, _cer_stat = edit_distance_with_stats(decoded_chars, ref_chars)
            _cer_len = len(ref_chars)

            if 'decoded_scores' in decoded and decoded['decoded_scores'] is not None:
                other = {k: v[i] for k, v in decoded['decoded_scores'].items()}
            else:
                other = {}

            if output_callback:
                output_callback(
                    uttid=uttid,
                    recognized=decoded_sentence,
                    original=ref_sentence,
                    wer=1.0*_wer_sum / _wer_len,
                    cer=1.0*_cer_sum / _cer_len,
                    wer_stat=_wer_stat,
                    cer_stat=_cer_stat,
                    text_loss=text_loss[i] if generate_data_losses else None,
                    other=other,
                    generated_loss=(
                        generated_loss[i] if generate_data_losses else None),
                    logits_text_diff=(
                        logits_diff[i] if generate_data_losses else None)
                    )
            wer_sum += _wer_sum
            cer_sum += _cer_sum
            wer_len += _wer_len
            cer_len += _cer_len
            len_ratio += len(decoded_chars) / len(ref_chars)
            sample_cnt += 1

    summary = {key: losses_stats[key].mean() for key in losses_stats}
    summary['WER'] = 1.0 * wer_sum / wer_len
    summary['CER'] = 1.0 * cer_sum / cer_len
    if logits_dumper:
        logits_dumper.end()
    summary['len_ratio'] = len_ratio / sample_cnt
    return summary


def evaluate_greedy(*args, **kwargs):
    kwargs['model_in_eval'] = True
    return do_evaluate(*args, **kwargs)


def evaluate_greedy_in_train_mode(*args, **kwargs):
    kwargs['model_in_eval'] = False
    return do_evaluate(*args, **kwargs)

def evaluate_speaker_classification(dataset, model, model_in_eval=True,
                                    progress_callback=None, **kwargs):
    dataset_length = len(dataset)

    if model_in_eval:
        model.eval()
    else:
        model.train()
    is_cuda = next(model.parameters()).is_cuda

    accuracy_sum = 0.
    accuracy_len = 0.

    for j, batch in enumerate(dataset):
        feature_lens = Variable(batch['features'][1])
        features = Variable(batch['features'][0])

        text_lens = Variable(batch['texts'][1])
        texts = Variable(batch['texts'][0])

        speakers = batch['spkids']

        ivectors = batch['ivectors']
        if ivectors is not None:
            ivectors = Variable(ivectors)
            if is_cuda:
                ivectors = ivectors.cuda()

        args = {}
        for key in batch:
            if key not in ['features', 'texts', 'spkids', 'uttids',
                           'ivectors']:
                args[key] = batch[key]

        if is_cuda:
            features = features.cuda()

        if progress_callback:
            progress_callback(j, dataset_length, features.size()[0])

        decoded = model.decode(
            features, feature_lens, speakers, texts, text_lens, **args)
        out = decoded['decoded']
        loss = decoded['loss']

        accuracy_sum -= sum([o == s for o, s in zip(out, speakers)])
        accuracy_len += len(out)

    summary = {'accuracy': accuracy_sum / accuracy_len}
    return summary


def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_mask(lengths, mask_length=None, batch_first=True):
    """Get mask that is 1 for sequences shorter than lengths and 0 otherwise.

    The mask is always on CPU, just like lengths.
    """
    if mask_length is None:
        mask_length = lengths.max()
    lengths = lengths.long()
    if batch_first:
        mask = torch.arange(mask_length) < lengths[:, None]
    else:
        mask = torch.arange(mask_length)[:, None] < lengths
    return Variable(mask.float())


def extract_modify_dict(modify_config):
    if modify_config is None:
        return {}
    even_list, odd_list = modify_config[::2], modify_config[1::2]
    if(len(even_list) != len(odd_list)):
        raise Exception(
            "Modify config list should have even number of elements")
    return dict(zip(even_list, odd_list))


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_num_vs_spk(speakers_filename):
    '''
    Returns pair consisting of:
    * dict (spk2num) mapping speaker name to number
    * list (num2spk) mapping number to speaker name
    '''
    num2spk = [line.split()[0] for line in open(speakers_filename, "r")]
    spk2num = {spkid: i for i, spkid in enumerate(num2spk)}
    return (spk2num, num2spk)

def get_accuracy(logits, labels):
    '''
    Both logits should be torch tensors, with following properties:
    * logits.size() = [num_batch, num_labels]
    * labels.size() = [num_batch]
    '''
    labels_predicted = torch.argmax(logits, dim=1)
    return torch.tensor(
        float((labels_predicted == labels).sum().item()) / len(labels))
