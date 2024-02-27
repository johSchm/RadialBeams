import tensorflow as tf
import numpy as np

from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import activations


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
