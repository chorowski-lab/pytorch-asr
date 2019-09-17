from functools import partial
import numpy as np
import random


class TextAugmenter(object):
    def __init__(self, config, characters, space_character=None):
        if config is None:
            config = {}
        self.augmenters = []
        self.characters = characters
        self.space_character = \
            space_character if config.pop('preserve_spaces', True) else None
        for augmenter, prob in config.items():
            if not isinstance(prob, list):
                prob = [prob]
            self.augmenters += [
                partial(getattr(self, '_'+augmenter), *prob)
            ]

    def _remove(self, prob, text):
        return [c for c in text if random.random() >= prob]

    def _swap(self, prob, text):
        for i in xrange(len(text)//2):
            if random.random() < prob:
                text[2*i], text[2*i+1] = text[2*i+1], text[2*i]
        return text

    def _replace(self, prob, text):
        for i in xrange(len(text)):
            if random.random() < prob:
                text[i] = random.choice(self.characters)
        return text

    def __apply(self, func, text):
        return [func(o) if o is not None else None for o in text]

    def __split(self, text):
        lists = []
        cur_list = []
        for elem in text:
            if elem != self.space_character:
                cur_list += [elem]
            else:
                if cur_list:
                    lists += [cur_list]
                    cur_list = []
                lists += [None]
        if cur_list:
            lists += [cur_list]
        return lists

    def __combine(self, texts):
        out = []
        for t in texts:
            if t is None:
                out += [self.space_character]
            else:
                out += t
        return out

    def __call__(self, text):
        if not self.augmenters:
            return text
        augmenters_i = range(len(self.augmenters))
        random.shuffle(augmenters_i)
        dtype = text.dtype
        text = text.tolist()
        texts = self.__split(text)
        for augmenter_i in augmenters_i:
            augmenter = self.augmenters[augmenter_i]
            texts = self.__apply(augmenter, texts)
        text = self.__combine(texts)
        return np.array(text, dtype=dtype)
