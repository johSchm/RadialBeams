import tensorflow as tf
import numpy as np
import math

from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import activations
from tensorflow.keras import layers, models


class ResNeXtBlock1D(tf.keras.Model):
    def __init__(self, n_filters: int, n_groups: int, l2_bias=None, l2_weight=None,
                 use_conv_bias=True, use_norm_bias=True, name=None, kernel_size=5):
        super(ResNeXtBlock1D, self).__init__(name=name)

        l2_bias = l2_bias if l2_bias is not None else 0.
        l2_weight = l2_weight if l2_weight is not None else 0.

        self.conv_block = models.Sequential([
            layers.Conv1D(n_filters // 2 if n_filters > 1 else n_filters, 1, groups=1,
                          padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2(l2=l2_bias),
                          use_bias=use_conv_bias, name="cv0"),
            layers.ELU(),
            layers.LayerNormalization(center=use_norm_bias, name="ln0"),

            layers.Conv1D(n_filters, kernel_size, groups=n_groups,
                          padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2(l2=l2_bias),
                          use_bias=use_conv_bias, name="cv1"),
            layers.ELU(),
            layers.LayerNormalization(center=use_norm_bias, name="ln1"),

            layers.Conv1D(n_filters, 1, groups=1,
                          padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2(l2=l2_bias),
                          use_bias=use_conv_bias, name="cv2"),
            layers.ELU(),
            layers.LayerNormalization(center=use_norm_bias, name="ln2"),
        ], name='main')
        self.residual_block = models.Sequential([
            layers.Conv1D(n_filters, kernel_size, groups=1,
                          padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2(l2=l2_bias),
                          use_bias=use_conv_bias, name="cv"),
            layers.LayerNormalization(center=use_norm_bias, name="ln")
        ], name='res')

    def call(self, inputs, training=None, **kwargs):
        x = self.conv_block(inputs)
        x_res = self.residual_block(inputs)
        return tf.nn.elu(tf.keras.layers.add([x, x_res]))


class PolarRegressor1D(tf.keras.Model):
    """ Using Cyclic Feature Encoding.
    """

    def __init__(self, n_beams, len_beam, n_filters=128, n_channels=3, l2_bias=0.01):
        super().__init__()
        # padding such that kernels of size 3 on the edges use cyclic padded values
        # padding = kernel_size - 1 using "same" padding
        self.receptive_field = 24  # +1
        self.padding = self.receptive_field // 2

        # This embeds each beam into a latent representation, mapping (C=3) -> (L)
        self.beam_encoding = None
        self.beam_encoder = models.Sequential([
            layers.InputLayer(input_shape=(n_beams + 2 * self.padding, len_beam * n_channels)),
            layers.Dense(units=n_filters)
        ], name='enc0')

        # This encodes the polar representation of the image (batch x len_beams x n_beams+padding x channels)
        # down to an energy map of shape (batch x len_beams x n_beams x 1), which preservers translation-equivariance.
        self.latent_polar_map = None
        self.latent_polar_encoder = models.Sequential([
            layers.InputLayer(input_shape=(n_beams + 2 * self.padding, n_filters)),

            ResNeXtBlock1D(n_filters=n_filters // 4, n_groups=4, l2_bias=.5, l2_weight=.5,
                           use_conv_bias=True, use_norm_bias=True, name='rb0'),
            ResNeXtBlock1D(n_filters=n_filters // 2, n_groups=8, l2_bias=0.1, l2_weight=0.1,
                           use_conv_bias=True, use_norm_bias=True, name='rb1'),
            ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.05, l2_weight=0.05,
                           use_conv_bias=True, use_norm_bias=True, name='rb2'),
            ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.05, l2_weight=0.05,
                           use_conv_bias=True, use_norm_bias=True, name='rb3'),
            ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.01, l2_weight=0.01,
                           use_conv_bias=True, use_norm_bias=True, name='rb4'),
            ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.01, l2_weight=0.01,
                           use_conv_bias=True, use_norm_bias=True, name='rb5'),
        ], name='enc1')

        # This maps the transposed energy map (batch x n_beams x len_beams) down to a radial energy over S^1
        # of shape (batch x n_beams), which preservers translation-equivariance.
        self.radial_energy = None
        self.radial_energy_encoder = models.Sequential([
            layers.InputLayer(input_shape=(n_beams, n_filters)),
            layers.Dense(n_filters),
            layers.Dense(1),
        ], name='enc2')

    def cyclic_beam_padding(self, x):
        return tf.concat([x[:, :, x.shape[2] - self.padding:], x, x[:, :, :self.padding]], axis=-2)

    def call(self, x):
        # pad the input sequence of radial beams (i.e., the polar representation)
        if self.padding > 0:
            x = self.cyclic_beam_padding(x)
        x = tf.transpose(x, (0, 2, 1, 3))
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1))
        self.beam_encoding = self.beam_encoder(x)
        self.latent_polar_map = self.latent_polar_encoder(self.beam_encoding)
        self.radial_energy = self.radial_energy_encoder(self.latent_polar_map)
        # log-probabilities
        return tf.nn.log_softmax(tf.squeeze(self.radial_energy, axis=-1), axis=-1)


class CircularPadding(layers.Layer):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def call(self, x):
        return tf.concat([x[:, :, x.shape[2] - self.padding:], x, x[:, :, :self.padding]], axis=-2)


class ResNeXtBlock(tf.keras.Model):
    def __init__(self, n_filters: int, n_groups: int, l2_bias=None, l2_weight=None, kernel_size=(3, 5),
                 use_conv_bias=True, use_norm_bias=True, name=None):
        super(ResNeXtBlock, self).__init__(name=name)

        l2_bias = l2_bias if l2_bias is not None else 0.
        l2_weight = l2_weight if l2_weight is not None else 0.

        self.conv_block = models.Sequential([
            layers.Conv2D(n_filters // 2 if n_filters > 1 else n_filters, (1, 1), groups=1,
                          padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2(l2=l2_bias),
                          use_bias=use_conv_bias, name="cv0"),
            layers.ELU(),
            layers.LayerNormalization(center=use_norm_bias, name="ln0"),

            CircularPadding((kernel_size[1] - 1) // 2),
            layers.Conv2D(n_filters, kernel_size, groups=n_groups,
                          padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2(l2=l2_bias),
                          use_bias=use_conv_bias, name="cv1"),
            layers.ELU(),
            layers.LayerNormalization(center=use_norm_bias, name="ln1"),

            layers.Conv2D(n_filters, (1, 1), groups=1,
                          padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2(l2=l2_bias),
                          use_bias=use_conv_bias, name="cv2"),
            layers.ELU(),
            layers.LayerNormalization(center=use_norm_bias, name="ln2"),
        ], name='main')
        self.residual_block = models.Sequential([
            CircularPadding((kernel_size[1] - 1) // 2),
            layers.Conv2D(n_filters, kernel_size, groups=1,
                          padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2(l2=l2_bias),
                          use_bias=use_conv_bias, name="cv"),
            layers.LayerNormalization(center=use_norm_bias, name="ln")
        ], name='res')

    def call(self, inputs, training=None, **kwargs):
        x = self.conv_block(inputs)
        x_res = self.residual_block(inputs)
        return tf.nn.elu(tf.keras.layers.add([x, x_res]))


class PolarRegressor2D(tf.keras.Model):
    """ Using Cyclic Feature Encoding.
    """

    def __init__(self, n_beams, len_beam, n_filters=128, n_channels=3, l2_bias=0.01):
        super().__init__()
        # This encodes the polar representation of the image (batch x len_beams x n_beams+padding x channels)
        # down to an energy map of shape (batch x len_beams x n_beams x 1), which preservers translation-equivariance.
        self.latent_polar_map = None
        self.latent_polar_encoder = models.Sequential([
            layers.InputLayer(input_shape=(len_beam, n_beams, n_channels)),

            ResNeXtBlock(n_filters=n_filters // 4, n_groups=4, l2_bias=.5, #l2_weight=.5,
                         use_conv_bias=True, use_norm_bias=True, name='rb0'),
            ResNeXtBlock(n_filters=n_filters // 2, n_groups=8, l2_bias=0.1, #l2_weight=0.1,
                         use_conv_bias=True, use_norm_bias=True, name='rb1'),
            ResNeXtBlock(n_filters=n_filters, n_groups=16, l2_bias=0.05, #l2_weight=0.05,
                         use_conv_bias=True, use_norm_bias=True, name='rb2'),
            ResNeXtBlock(n_filters=n_filters, n_groups=16, l2_bias=0.05, #l2_weight=0.05,
                         use_conv_bias=True, use_norm_bias=True, name='rb3'),
            ResNeXtBlock(n_filters=n_filters, n_groups=16, l2_bias=0.01, #l2_weight=0.01,
                         use_conv_bias=True, use_norm_bias=True, name='rb4'),
            ResNeXtBlock(n_filters=n_filters, n_groups=16, l2_bias=0.01, #l2_weight=0.01,
                         use_conv_bias=True, use_norm_bias=True, name='rb5'),
            ResNeXtBlock(n_filters=n_filters, n_groups=16, l2_bias=0.01,  # l2_weight=0.01,
                         use_conv_bias=True, use_norm_bias=True, name='rb6'),
            # ResNeXtBlock(n_filters=n_filters, n_groups=16, l2_bias=0.01,  # l2_weight=0.01,
            #              use_conv_bias=True, use_norm_bias=True, name='rb7'),
            # required if beam cut-off is 3 not 5
            # layers.Conv2D(n_filters, (3, 1), groups=1,
            #               padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(),
            #               kernel_regularizer=tf.keras.regularizers.l2(0.01),
            #               bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2(l2=l2_bias),
            #               use_bias=True, name="final"),
        ], name='enc0')

        self.latent_radial_energy = None
        self.latent_radial_energy_encoder = models.Sequential([
            layers.Conv2D(n_filters, (9, 1), activation='elu', padding='valid',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2()),
            layers.Conv2D(n_filters, (9, 1), activation='elu', padding='valid',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2()),
            layers.LayerNormalization(center=True),

            layers.Conv2D(n_filters, (9, 1), activation='elu', padding='valid',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2()),
            layers.Conv2D(n_filters, (9, 1), activation='elu', padding='valid',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          bias_initializer='zeros', bias_regularizer=tf.keras.regularizers.l2()),
            layers.LayerNormalization(center=True),
            # layers.InputLayer(input_shape=(n_beams, len_beam - 14, n_filters)),
            # ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.01, kernel_size=5,
            #                use_conv_bias=True, use_norm_bias=True, name='rb6'),
            # ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.01, kernel_size=5,
            #                use_conv_bias=True, use_norm_bias=True, name='rb7'),
            # ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.01, kernel_size=7,
            #                use_conv_bias=True, use_norm_bias=True, name='rb8'),
            # ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.01, kernel_size=7,
            #                use_conv_bias=True, use_norm_bias=True, name='rb9'),
            # ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.01, kernel_size=7,
            #                use_conv_bias=True, use_norm_bias=True, name='rb10'),
            # ResNeXtBlock1D(n_filters=n_filters, n_groups=16, l2_bias=0.01, kernel_size=7,
            #                use_conv_bias=True, use_norm_bias=True, name='rb11'),
        ], name='enc1')

        # This maps the transposed energy map (batch x n_beams x len_beams) down to a radial energy over S^1
        # of shape (batch x n_beams), which preservers translation-equivariance.
        self.radial_energy = None
        self.radial_energy_encoder = models.Sequential([
            layers.InputLayer(input_shape=(n_beams, n_filters)),
            layers.Dense(n_filters),
            layers.Dense(1),
        ], name='enc2')

    @staticmethod
    def gaussian_smoothing(logits, size=8, sigma=1.):
        x = tf.linspace(-1, 1, size)
        # \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
        kernel = (tf.cast(1 / (sigma * tf.math.sqrt(2 * math.pi)), float)
                  * tf.cast(tf.math.exp(-0.5 * (x / sigma) ** 2), float))
        logits = tf.concat([logits[:, logits.shape[1] - (size - 1) // 2:], logits, logits[:, :(size - 1) // 2]], axis=1)
        # kernel: [filter_width, in_channels, out_channels]
        logits = tf.nn.conv1d(logits[..., None], kernel[:, None, None], stride=1, padding="VALID", data_format='NWC')
        return tf.squeeze(logits, axis=-1)

    def call(self, x, ema=False):
        # compute the energy map of the input
        self.latent_polar_map = self.latent_polar_encoder(x)
        # x = tf.transpose(self.latent_polar_map, (0,2,1,3))
        self.latent_radial_energy = self.latent_radial_energy_encoder(self.latent_polar_map)
        # from the energy map, we estimate an energy function over S^1
        # z = tf.transpose(self.energy_map, (0, 2, 1, 3)) # tf.squeeze(self.energy_map, axis=-1)
        # x = tf.squeeze(self.latent_radial_energy)
        x = tf.squeeze(self.latent_radial_energy)
        self.radial_energy = self.radial_energy_encoder(x)
        # we convolve this radial energy by a learned kernel
        # z = self.angle_encoder(self.radial_energy)
        # normalise the angle vector
        # return z / tf.norm(z, axis=-1, keepdims=True)
        # log-probabilities
        x = tf.squeeze(self.radial_energy, axis=-1)
        x = tf.nn.log_softmax(x, axis=-1)
        if ema:
            x = self.gaussian_smoothing(x, size=9, sigma=1.)
        return x


class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(tf.keras.layers.Conv2D(filters=self.group_out_num,
                                                         kernel_size=kernel_size,
                                                         strides=strides,
                                                         padding=padding,
                                                         data_format=data_format,
                                                         dilation_rate=dilation_rate,
                                                         activation=activations.get(activation),
                                                         use_bias=use_bias,
                                                         kernel_initializer=initializers.get(kernel_initializer),
                                                         bias_initializer=initializers.get(bias_initializer),
                                                         kernel_regularizer=regularizers.get(kernel_regularizer),
                                                         bias_regularizer=regularizers.get(bias_regularizer),
                                                         activity_regularizer=regularizers.get(activity_regularizer),
                                                         kernel_constraint=constraints.get(kernel_constraint),
                                                         bias_constraint=constraints.get(bias_constraint),
                                                         **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i * self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out


class ResNeXt_BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(ResNeXt_BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters//2 if filters > 1 else filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.group_conv = GroupConv2D(input_channels=filters//2 if filters > 1 else filters,
                                      output_channels=filters//2 if filters > 1 else filters,
                                      kernel_size=(3, 3),
                                      strides=strides,
                                      padding="same",
                                      groups=groups)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.shortcut_conv = tf.keras.layers.Conv2D(filters=filters,
                                                    kernel_size=(1, 1),
                                                    strides=strides,
                                                    padding="same")
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.gelu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.gelu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = tf.nn.gelu(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.gelu(tf.keras.layers.add([x, shortcut]))
        return output


class MNISTClassifier(tf.keras.layers.Layer):

    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.linear1 = tf.keras.layers.Dense(28 * 28)
        self.linear2 = tf.keras.layers.Dense(256, activation='leaky_relu')
        self.linear3 = tf.keras.layers.Dense(9, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return self.linear3(x)


class GraphConvolutionLayer(tf.keras.layers.Layer):

    def __init__(self, units, A, activation=tf.identity, rate=0.0, l2=0.0):
        super(GraphConvolutionLayer, self).__init__()
        self.activation = activation
        self.units = units
        self.rate = rate
        self.l2 = l2
        self.A = A

    def build(self, input_shape):
        self.W = self.add_weight(
          shape=(input_shape[-1], self.units),
          dtype=self.dtype,
          name='gnn_weights',
          initializer='glorot_uniform',
          regularizer=tf.keras.regularizers.l2(self.l2)
        )

    def call(self, X):
        """
        input (batch x vector x hidden)
        output (batch x vector x hidden)
        """
        X = tf.nn.dropout(X, self.rate)
        X = self.A @ X @ self.W
        return self.activation(X)


def wheel_graph_adjacency_matrix(n_vectors):
    adjacency = np.zeros([n_vectors + 1, n_vectors + 1])
    adjacency[0, 1] = 1
    adjacency[0, n_vectors - 1] = 1
    for i in range(1, n_vectors - 1):
        adjacency[i, i-1] = 1
        adjacency[i, -1] = 1
    adjacency[n_vectors - 1, n_vectors - 2] = 1
    adjacency[n_vectors - 1, 0] = 1
    return tf.cast(adjacency, tf.float32)


def circular_graph_adjacency_matrix(n_vectors):
    adjacency = np.zeros([n_vectors, n_vectors])
    adjacency[0, 1] = 1
    adjacency[0, n_vectors - 1] = 1
    for i in range(1, n_vectors - 1):
        adjacency[i, i-1] = 1
    adjacency[n_vectors - 1, n_vectors - 2] = 1
    adjacency[n_vectors - 1, 0] = 1
    return adjacency


def angle_matrix(n_vectors: int) -> tf.Tensor:
    """ Returns triangular matrix degrading to the center
    """
    matrix = tf.fill([n_vectors, n_vectors], float(n_vectors))
    for i in tf.range(n_vectors):
        matrix -= tf.linalg.band_part(tf.ones((n_vectors, n_vectors)), i, -1)
        matrix += tf.linalg.band_part(tf.ones((n_vectors, n_vectors)), 0, i)
    return matrix - float(n_vectors) * tf.eye(n_vectors)


def toeplitz_extractor(n_vectors: int) -> tf.Tensor:
    return tf.cast(
        [tf.roll(tf.eye(n_vectors), shift=i, axis=0) for i in tf.range(n_vectors)],
        tf.float32)


class BeamEncoder(tf.keras.layers.Layer):

    def __init__(self, hidden=128, target_size=28, activation=tf.nn.leaky_relu, n_pixels=14, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.activation = activation
        self.target_size = target_size
        self.n_pixels = n_pixels
        self.w_init = tf.keras.initializers.HeNormal()
        self.b_init = tf.constant_initializer(0.01)
        self.proxkernels = []
        self.tempkernels = []

    def build(self, input_shape):
        # general output: (batch x 1 x vec x 0 x n-1 x H/8)
        self.proxkernels = [
            # (batch x augmentation=1 x vec x proximity=3 x pixels=14 x 0) -> (batch x 1 x vec x 3-1 x n-1 x H/8)
            tf.keras.layers.Conv2D(self.hidden // 8, (3, 3), data_format="channels_last",
                                   activation=self.activation, padding='valid'),
        ]

        # eg fashion mnist without margin
        if self.n_pixels == 8:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 8-1 x H/8) -> (batch x 1 x vec x 0 x 7-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 4, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 4 x H/8) -> (batch x 1 x vec x 0 x 4-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 3, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
            ]

        # eg fashion mnist
        elif self.n_pixels == 14:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 14-1 x H/8) -> (batch x 1 x vec x 0 x 14-5 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 4, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 14-5 x H/8) -> (batch x 1 x vec x 0 x 14-8 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 14-8 x H/8) -> (batch x 1 x vec x 0 x 14-11 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 14-11 x H/4) -> (batch x 1 x vec x 0 x 14-14 x H/1)
                tf.keras.layers.Conv1D(self.hidden, 3, activation=self.activation,
                                       padding='valid', data_format="channels_last")
            ]

        # eg fashion mnist with margin
        elif self.n_pixels == 20:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 20-1 x H/8) -> (batch x 1 x vec x 0 x 19-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 4, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 20-1 x H/8) -> (batch x 1 x vec x 0 x 19-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 5, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 20-1 x H/8) -> (batch x 1 x vec x 0 x 19-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 6, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 20-1 x H/8) -> (batch x 1 x vec x 0 x 19-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 6, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
            ]

        # eg cifar without margin
        elif self.n_pixels == 9:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 9-1 x H/8) -> (batch x 1 x vec x 0 x 8-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 3, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 4 x H/8) -> (batch x 1 x vec x 0 x 4-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 5, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
            ]

        # eg cifar
        elif self.n_pixels == 16:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 16-1 x H/8) -> (batch x 1 x vec x 0 x 16-5 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 4, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 16-5 x H/8) -> (batch x 1 x vec x 0 x 16-8 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 16-8 x H/8) -> (batch x 1 x vec x 0 x 16-11 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 16-11 x H/4) -> (batch x 1 x vec x 0 x 16-14 x H/1)
                tf.keras.layers.Conv1D(self.hidden // 1, 4, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 16-14 x H/4) -> (batch x 1 x vec x 0 x 16-16 x H/1)
                tf.keras.layers.Conv1D(self.hidden // 1, 2, activation=self.activation,
                                       padding='valid', data_format="channels_last")
            ]

        # eg cifar with margin
        elif self.n_pixels == 23:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 23-1 x H/8) -> (batch x 1 x vec x 0 x 22-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 5, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 18 x H/8) -> (batch x 1 x vec x 0 x 18-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 5, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 14 x H/8) -> (batch x 1 x vec x 0 x 14-5 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 6, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 9 x H/4) -> (batch x 1 x vec x 0 x 9-5 x H/1)
                tf.keras.layers.Conv1D(self.hidden // 2, 6, activation=self.activation,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 4 x H/4) -> (batch x 1 x vec x 0 x 4-3 x H/1)
                tf.keras.layers.Conv1D(self.hidden, 3, activation=self.activation,
                                       padding='valid', data_format="channels_last")
            ]

        elif self.n_pixels == 32:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 32-1 x H/8) -> (batch x 1 x vec x 0 x 32-6/1=13 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 5, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13 x H/8) -> (batch x 1 x vec x 0 x 13-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-3 x H/8) -> (batch x 1 x vec x 0 x 13-6 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-6 x H/8) -> (batch x 1 x vec x 0 x 13-9 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-9 x H/8) -> (batch x 1 x vec x 0 x 13-11 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 1, 3, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-11 x H/8) -> (batch x 1 x vec x 0 x 13-13 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 1, 2, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
            ]

        # eg coil100 without margin
        elif self.n_pixels == 37:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 37-1 x H/8) -> (batch x 1 x vec x 0 x 37-3/2 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 4, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 17 x H/8) -> (batch x 1 x vec x 0 x 17-3/2 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 4, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 7 x H/8) -> (batch x 1 x vec x 0 x 7-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 4 x H/8) -> (batch x 1 x vec x 0 x 4-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
            ]

        # eg coil100
        elif self.n_pixels == 64:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 64-1 x H/8) -> (batch x 1 x vec x 0 x 64-6/1=29 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 5, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 29 x H/8) -> (batch x 1 x vec x 0 x 29-3/1=13 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 4, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13 x H/8) -> (batch x 1 x vec x 0 x 13-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-3 x H/8) -> (batch x 1 x vec x 0 x 13-6 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-6 x H/8) -> (batch x 1 x vec x 0 x 13-9 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-9 x H/8) -> (batch x 1 x vec x 0 x 13-11 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 1, 3, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-11 x H/8) -> (batch x 1 x vec x 0 x 13-13 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 1, 2, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
            ]

        # eg coil100 with margin
        elif self.n_pixels == 91:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 91-1 x H/8) -> (batch x 1 x vec x 0 x 90-4/2 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 5, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 43 x H/8) -> (batch x 1 x vec x 0 x 43-3/2 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 4, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 20 x H/8) -> (batch x 1 x vec x 0 x 20-5 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 6, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 15 x H/8) -> (batch x 1 x vec x 0 x 15-5 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 6, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 10 x H/8) -> (batch x 1 x vec x 0 x 10-5 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 6, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 5 x H/8) -> (batch x 1 x vec x 0 x 5-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 5, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
            ]

        # eg lfw without margin
        elif self.n_pixels == 99:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 99-1 x H/8) -> (batch x 1 x vec x 0 x 98-4/2 x H/8)
                tf.keras.layers.Conv1D(self.hidden // 4, 5, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 47 x H/8) -> (batch x 1 x vec x 0 x 47-3/2 x H/8)
                tf.keras.layers.Conv1D(self.hidden // 4, 4, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 22 x H/8) -> (batch x 1 x vec x 0 x 22-2/2 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 3, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 10 x H/8) -> (batch x 1 x vec x 0 x 10-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 5, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 6 x H/8) -> (batch x 1 x vec x 0 x 6-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 3 x H/8) -> (batch x 1 x vec x 0 x 3-2 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 3, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
            ]

        # eg lfw
        elif self.n_pixels == 125:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 125-1 x H/8) -> (batch x 1 x vec x 0 x 125-5/1 x H/8)
                tf.keras.layers.Conv1D(self.hidden // 4, 4, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 60 x H/8) -> (batch x 1 x vec x 0 x 60-1/1 x H/8)
                tf.keras.layers.Conv1D(self.hidden // 4, 3, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 29 x H/8) -> (batch x 1 x vec x 0 x 29-3/1=13 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 4, 4, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13 x H/8) -> (batch x 1 x vec x 0 x 13-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-3 x H/8) -> (batch x 1 x vec x 0 x 13-6 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-6 x H/8) -> (batch x 1 x vec x 0 x 13-9 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-9 x H/8) -> (batch x 1 x vec x 0 x 13-11 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 1, 3, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 13-11 x H/8) -> (batch x 1 x vec x 0 x 13-13 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 1, 2, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
            ]

        # eg lfw with margin
        elif self.n_pixels == 177:
            self.tempkernels = [
                # (batch x 1 x vec x 0 x 177-1 x H/8) -> (batch x 1 x vec x 0 x 176-4/2 x H/8)
                tf.keras.layers.Conv1D(self.hidden // 4, 5, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 86 x H/8) -> (batch x 1 x vec x 0 x 86-4/2 x H/8)
                tf.keras.layers.Conv1D(self.hidden // 4, 5, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 41 x H/8) -> (batch x 1 x vec x 0 x 41-3/2 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 19 x H/8) -> (batch x 1 x vec x 0 x 19-3/2 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 4, activation=self.activation, strides=2,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 8 x H/8) -> (batch x 1 x vec x 0 x 8-4 x H/4)
                tf.keras.layers.Conv1D(self.hidden // 2, 5, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
                # (batch x 1 x vec x 0 x 4 x H/8) -> (batch x 1 x vec x 0 x 4-3 x H/4)
                tf.keras.layers.Conv1D(self.hidden, 4, activation=self.activation, strides=1,
                                       padding='valid', data_format="channels_last"),
            ]

        else:
            raise ValueError('Other n pixels not yet supported!')

    def call(self, inputs, training=None):
        """ convolution compressing each vector to a lower dimensional representation
        starting with kernel to aggregate the proximity and spatial information
        secondly, aggregate the spatial information fully via 1D spatial convolution
        :param inputs:
        :param training:
        :return:
        """
        x = inputs
        for kernel in self.proxkernels:
            x = kernel(x)
        for kernel in self.tempkernels:
            x = kernel(x)
        x = tf.reshape(x, [tf.shape(x)[0], 2, self.target_size, self.hidden])
        return x

    def get_config(self):
        config = {
            'hidden': self.hidden,
            'activation': self.activation,
            'target_size': self.target_size,
            'n_pixels': self.n_pixels,
            'w_init': self.w_init,
            'b_init': self.b_init
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class BIC(tf.keras.models.Model):

    def __init__(self, hidden=128, activation=tf.nn.leaky_relu, lstm_layers=3,
                 l2_regularization=0.0, edge_factor=0.5, gcn_layers=3, dropout=0.0,
                 n_beams=28, pixel_count_per_beam=14,
                 context=True, **kwargs):
        super(BIC, self).__init__(**kwargs)
        self.context = context
        self.l2_regularization = l2_regularization
        self.regularizer = tf.keras.regularizers.L2(l2_regularization)
        self.n_beams = n_beams
        self.pixel_count_per_beam = pixel_count_per_beam
        self.hidden = hidden
        self.activation = activation
        self.edge_factor = edge_factor
        self.gcn_layers = gcn_layers
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.w_init = tf.keras.initializers.HeNormal()
        self.b_init = tf.constant_initializer(0.01)
        self.adjacency = None
        self.beamenc = None
        self.extractor = None
        self.cxtenc = None
        self.tempkernels = []
        self.gcn = []
        self.lstm = []
        self.mlp = []

    def functional_compile(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def build(self, input_shape):
        self.beamenc = BeamEncoder(hidden=self.hidden, target_size=self.n_beams,
                                   activation=self.activation, n_pixels=self.pixel_count_per_beam)
        self.adjacency = wheel_graph_adjacency_matrix(self.n_beams) * self.edge_factor
        self.extractor = toeplitz_extractor(self.n_beams)
        self.gcn = [
            GraphConvolutionLayer(self.hidden, self.adjacency, activation=self.activation,
                                  rate=self.dropout, l2=self.l2_regularization)
            for _ in tf.range(self.gcn_layers)
        ]
        self.lstm = [
            tf.keras.layers.LSTM(self.hidden, return_sequences=False if tf.equal(i, self.lstm_layers - 1) else True,
                                 name='lstm{}'.format(i))
            for i in tf.range(self.lstm_layers)
        ]
        self.mlp = [
            tf.keras.layers.Dense(self.hidden // 2,
                                  kernel_initializer=self.w_init, bias_initializer=self.b_init),
            tf.keras.layers.Dense(self.hidden // 4,
                                  kernel_initializer=self.w_init, bias_initializer=self.b_init),
            tf.keras.layers.Dense(2, #activation=tf.nn.tanh,
                                  kernel_initializer=self.w_init, bias_initializer=self.b_init)
        ]

    # @tf.function
    def call(self, inputs, training=None):
        """ convolution compressing each vector to a lower dimensional representation
        starting with kernel to aggregate the proximity and spatial information
        secondly, aggregate the spatial information fully via 1D spatial convolution
        :param inputs:
        :param training:
        :return:
        """
        batch_dim = tf.shape(inputs)[0]
        beamencoding = self.beamenc(inputs)

        # encode the neighbor / spatial relationships via a GCN (batch x 1 x vec x hidden)
        # important to encode neighborhood, since pure black vectors might appear multiple times in an image
        # context encoder
        ctx = tf.constant(0)
        if self.context:
            # init the context node with zeros
            ctx = tf.concat([beamencoding, tf.zeros([batch_dim, 2, 1, tf.shape(beamencoding)[-1]])], axis=-2)
            for i in range(len(self.gcn)):
                ctx = self.gcn[i](ctx)
            beamencoding += ctx[..., :-1, :]
            ctx = ctx[..., -1, :]

        # split (batch x 1 x vector x hidden) -> (batch x 0 x vector x hidden) x2
        beamencoding_zero, beamencoding_theta = tf.split(beamencoding, num_or_size_splits=2, axis=1)

        # reshape to (batch x 0 x vector x hidden) and (batch x vector x 0 x hidden)
        # distance matrix | (batch x 0 x vector x hidden) - (batch x vector x 0 x hidden) | = (batch x vector x vector)
        beamencoding_zero = tf.reshape(beamencoding_zero, [batch_dim, self.n_beams, self.hidden])
        beamencoding_theta = tf.reshape(beamencoding_theta, [batch_dim, self.n_beams, self.hidden])

        # is comparing the orientation even better since this leaves the magnitude to be used for the angle decoder
        # the higher the closer the vectors
        similarity = tf.reduce_sum(
            (tf.expand_dims(beamencoding_zero, 1) - tf.expand_dims(beamencoding_theta, 2)) ** 2, -1)
        similarity = 1 / (1 + similarity)

        # reshape back to (batch x vector x vector)
        prior = tf.reshape(similarity, [batch_dim, self.n_beams, self.n_beams])
        angle_energy = prior

        # Hadamard product with shifted Diagonal to extract diagonals from the masked Toeplitz Distance Matrix
        # extractor shape (0 x vector x vector x vector)
        prior = prior[:, None, ...] * self.extractor[None, ...]

        # sum together the elements in the matrix (ie distances) (all others are zero) -> (batch x vector)
        # crucial that permutation invariant op used since order on the diagonal does not matter
        # mean instead of sum, since sum leads to over confident distributions
        # however, since the rest but the shifted diagonal are zero, the mean is quiet small
        prior = tf.reduce_sum(prior, axis=(-1, -2))

        # # distribution over (batch x vector) which represents the shift
        prior = tf.nn.softmax(prior, axis=-1)

        # additionally for the deployment phase, we use a second prediction task ie predicting the
        # canonicalization vector 0 (upper left corner) for reference to be able to turn a single image back
        unit_vec = beamencoding_theta + ctx[:, 1, None, :] if self.context else beamencoding_theta
        unit_vec = tf.reshape(unit_vec, [batch_dim, self.n_beams, self.hidden])

        # the RNN aims to encode the positional information of the vectors (ordering)
        # which gives raise to the angle of rotation
        for i in range(len(self.lstm)):
            unit_vec = self.lstm[i](unit_vec)
        unit_vec = tf.reshape(unit_vec, [batch_dim, self.hidden])
        rnn_encoding = unit_vec

        # mlp decoder
        for i in range(len(self.mlp)):
            unit_vec = self.mlp[i](unit_vec)

        unit_vec /= tf.math.sqrt((unit_vec[:, 0] - 0) ** 2 + (unit_vec[:, 1] - 0) ** 2)[:, None]

        return prior, unit_vec, beamencoding, ctx, similarity, \
               beamencoding_zero, beamencoding_theta, angle_energy, rnn_encoding

    def get_config(self):
        config = {
            'hidden': self.hidden,
            'activation': self.activation,
            'w_init': self.w_init,
            'b_init': self.b_init
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class Block(tf.keras.models.Sequential):
    def __init__(self,n,m):
        super().__init__()
        for i in range(m):
            self.add(tf.keras.layers.Conv2D(filters = n, kernel_size=(3,3),
                                            strides=(1,1),padding = 'same',activation = "relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))


class Dense(tf.keras.models.Sequential):
    def __init__(self,n,m=2):
        super().__init__()
        for i in range(m):
            self.add(tf.keras.layers.Dense(units = n, activation = "relu"))


class VGG11(tf.keras.models.Sequential):
    def __init__(self, input_shape, classes, filters = 64):
        super().__init__()
        self.add(tf.keras.layers.InputLayer(input_shape = input_shape))

        # Backbone
        self.add(Block(n = filters * 1, m = 1))
        self.add(Block(n = filters * 2, m = 1))
        self.add(Block(n = filters * 4, m = 2))
        self.add(Block(n = filters * 8, m = 2))
        self.add(Block(n = filters * 8, m = 2))

        # top
        self.add(tf.keras.layers.Flatten())
        self.add(Dense(n = filters * 64))
        self.add(tf.keras.layers.Dense(units = classes,activation = "softmax"))
