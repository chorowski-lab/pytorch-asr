#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from torch import nn
import itertools

from att_speech import utils
from att_speech.modules import decoders


class SpeechModel(nn.Module):
    def __init__(self, encoder, decoder, sample_batch, num_classes, vocabulary,
                 **kwargs):
        super(SpeechModel, self).__init__(**kwargs)
        self.encoder = utils.contruct_from_kwargs(
                encoder, 'att_speech.modules.encoders',
                {'sample_batch': sample_batch})

        sample_batch["features"], sample_batch["features_lens"] = (
            self.encoder(**sample_batch))

        self.decoder = utils.contruct_from_kwargs(
                decoder, 'att_speech.modules.decoders',
                {'sample_batch': sample_batch,
                 'num_classes': num_classes,
                 'vocabulary': vocabulary})

    def load_state(self, state_dict, strict=True):
        self.load_state_dict(state_dict, strict)

    def forward(self, features, feature_lens, spkids, texts, text_lens,
                ivectors=None, **kwargs):
        """
        Compute the training loss

        Args:
            features: float32 tensor of size (bs x t x f x c)
            feature_lens: int64 tensor of size (bs)
            spkids: list of length bs with speakers id
            texts: float32 tensor of size (bs x l)
            text_lens: int64 tensor of size (bs)
            ivectors: ivectors (bs x dim)

        Returns:
            scalar loss
        """
        encoded, encoded_lens = self.encoder(features, feature_lens, spkids,
                                             ivectors, **kwargs)
        loss = self.decoder(encoded, encoded_lens, texts, text_lens,
                            spkids=spkids, **kwargs)
        return loss

    def decode(self, features, feature_lens, speakers,
               texts=None, text_lens=None, encoder_args=None,
               decoder_args=None, ivectors=None, **kwargs):
        """
        Decode input data and optionally return its loss

        Args:
            features: float32 tensor of size (bs x t x f x c)
            feature_lens: int64 tensor of size (bs)
            speakers: list of length bs with speakers id
            texts: float32 tensor of size (bs x l), optional
            text_lens: int64 tensor of size (bs), optional
            ivectors: ivectors

        Returns:
            #bs strings containing recognized text and if both
            texts and text_lens are present - scalar loss
        """
        encoder_args = {} if encoder_args is None else encoder_args
        decoder_args = {} if decoder_args is None else decoder_args

        encoder_args.update(kwargs)
        decoder_args.update(kwargs)

        with torch.no_grad():
            encoded, encoded_lens = self.encoder(
                features, feature_lens, speakers, ivectors, **encoder_args)
            output = self.decoder.decode(
                encoded, encoded_lens, texts,
                text_lens, spkids=speakers, **decoder_args)
        return output

    def get_parameters_for_optimizer(self):
        return itertools.chain(self.encoder.get_parameters_for_optimizer(),
                               self.decoder.parameters())
