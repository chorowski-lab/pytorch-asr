from collections import OrderedDict
from torch import nn

from base_encoder import BaseEncoder
from encoder_utils import (BatchRNN, Normalization,
                           SequentialWithOptionalAttributes)

##############################################################################
# Deep speech implementation based on
# https://github.com/SeanNaren/deepspeech.pytorch
##############################################################################


class DeepSpeech2(BaseEncoder):
    """
    Deep Speech 2 implementation
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 sample_batch,
                 conv_normalization='batch_norm',
                 conv_strides=[[2, 2], [2, 1]],
                 conv_kernel_sizes=[[41, 11], [21, 11]],
                 conv_num_features=[32, 32],
                 rnn_hidden_size=768,
                 rnn_nb_layers=5,
                 rnn_projection_size=0,
                 rnn_type=nn.LSTM,
                 rnn_dropout=0.0,
                 rnn_residual=False,
                 rnn_normalization='batch_norm',
                 rnn_subsample=None, # [x, y] == 2-fold subsample after x and yth layer
                 **kwargs):
        super(DeepSpeech2, self).__init__(**kwargs)

        self.makeConv(sample_batch, conv_strides, conv_kernel_sizes,
                      conv_num_features, conv_normalization)

        # Compute output size of self.conv
        features = sample_batch['features']
        features = features.permute(0, 3, 1, 2)
        features = self.extendFeatures(features=features)

        after_conv_size = self.conv.forward(features).size()
        self.rnn_input_size = self.computeRnnInputSize(
            sample_batch, after_conv_size)

        self.makeRnn(rnn_hidden_size, rnn_nb_layers, rnn_projection_size,
                     rnn_type, rnn_dropout, rnn_residual, rnn_normalization,
                     rnn_subsample)

    def makeConv(self, sample_batch, conv_strides, conv_kernel_sizes,
                 conv_num_features, normalization):
        num_channels = sample_batch['features'].size()[3]

        conv_padding = int(0.5 * (
            conv_kernel_sizes[1][0] - 1 +
            conv_strides[0][0] * (conv_kernel_sizes[0][0] - 1)))
        self.conv_cumative_stride = conv_strides[0][0] * conv_strides[1][0]

        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, conv_num_features[0],
                      kernel_size=conv_kernel_sizes[0],
                      stride=conv_strides[0], padding=(conv_padding, 0)),
            Normalization(normalization, 2, conv_num_features[0]),
            nn.Hardtanh(0, 20, inplace=True),

            nn.Conv2d(conv_num_features[0], conv_num_features[1],
                      kernel_size=conv_kernel_sizes[1],
                      stride=conv_strides[1]),
            Normalization(normalization, 2, conv_num_features[1]),
            nn.Hardtanh(0, 20, inplace=True)
        )

    def makeRnn(self, rnn_hidden_size, rnn_nb_layers, rnn_projection_size,
                rnn_type, rnn_dropout, rnn_residual, normalization,
                rnn_subsample):
        if rnn_subsample is None:
            rnn_subsample = []
        if rnn_dropout > 0.0:
            rnn_dropout = nn.modules.Dropout(p=rnn_dropout)
        else:
            rnn_dropout = None

        rnns = []
        rnn = BatchRNN(input_size=self.rnn_input_size,
                       hidden_size=rnn_hidden_size,
                       rnn_type=rnn_type, bidirectional=True,
                       packed_data=True, normalization=None,
                       projection_size=rnn_projection_size,
                       subsample=(0 in rnn_subsample))
        rnns.append(('0', rnn))
        for i in range(rnn_nb_layers - 1):
            rnn = BatchRNN(input_size=(rnn_hidden_size
                                       if rnn_projection_size == 0
                                       else rnn_projection_size),
                           projection_size=rnn_projection_size,
                           hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=True, packed_data=True,
                           normalization=normalization,
                           residual=rnn_residual,
                           subsample=(i+1 in rnn_subsample))
            if rnn_dropout:
                rnns.append(('{}_dropout'.format(i+1), rnn_dropout))
            rnns.append(('{}'.format(i + 1), rnn))
        self.rnns = SequentialWithOptionalAttributes(OrderedDict(rnns))

    def lenOfCharacteristicVectors(self):
        return 0

    def computeRnnInputSize(self, sample_batch, after_conv_size):
        rnn_input_size = after_conv_size[1] * (
            after_conv_size[3] + self.lenOfCharacteristicVectors())
        return rnn_input_size

    def extendFeatures(self, **kwargs):
        return kwargs["features"]

    def extendFeaturesBeforeConv(self, **kwargs):
        return self.extendFeatures(**kwargs)

    def extendFeaturesAfterConv(self, **kwargs):
        return self.extendFeatures(**kwargs)

    def forward(self, features, features_lengths, spkids, ivectors=None,
                characteristic_vectors=None, **kwargs):
        # bs x t x f x c -> bs x c x t x f
        features = features.permute(0, 3, 1, 2)

        features = self.extendFeaturesBeforeConv(
            features=features,
            characteristic_vectors=characteristic_vectors,
            ivectors=ivectors)

        features = self.conv(features)

        features = self.extendFeaturesAfterConv(
            features=features,
            characteristic_vectors=characteristic_vectors,
            ivectors=ivectors)

        (batch_size, unused_num_channels, num_timestp, unused_num_features
         ) = features.size()

        # bs x c x t x f -> t x bs x c x f -> t x bs x (c x f)
        features = features.permute(2, 0, 1, 3).contiguous()
        features = features.view(num_timestp, batch_size, -1).contiguous()

        features_lengths = ((
            features_lengths + self.conv_cumative_stride - 1
            ) / self.conv_cumative_stride).int()
        assert features_lengths[0] == features.size()[0]

        return self.forwardRnn(features, features_lengths, num_timestp, spkids)

    def forwardRnn(self, features, features_lengths, num_timestp, spkids):
        features = nn.utils.rnn.pack_padded_sequence(
            features, features_lengths.data.numpy())
        features = self.rnns(features, spkids)
        return nn.utils.rnn.pad_packed_sequence(features)
