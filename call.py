import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(3)

import tensorflow as tf
print('TF Version: ' + str(tf.__version__))
physical_devices = tf.config.list_physical_devices('GPU')
print('GPUs: ' + str(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from src.learning import training
from src.model import BIC
from src.parsing import load_dataset
from src.parsing import preprocess
from src.parsing import instantiate_radial_vectors
from datetime import datetime
import argparse


def run(n_iter=4*1024, batch_size=1*128, beam_set_size=8, name='', context=True, prior='off',
        dataset='coil100', splits=[0.8, 0.1, 0.1], partial=None, continuous=False,
        learning_rate=0.0001, gcn_layers=3, beam_length=64, hidden_size=128):

    margin_padding = math.ceil(beam_length * (math.sqrt(2) - 1))

    dataset = load_dataset(dataset, partial=partial)

    n_train = int(splits[0] * float(dataset.cardinality()))
    n_val = int(splits[1] * float(dataset.cardinality()))
    n_test = int(splits[2] * float(dataset.cardinality()))

    train_dataset = dataset.take(n_train)
    val_dataset = dataset.skip(n_train).take(n_val)
    test_dataset = dataset.skip(n_train).skip(n_val).take(n_train)

    print('Target train size {0}, actual size {1}.'.format(n_train, train_dataset.cardinality()))
    print('Target val size {0}, actual size {1}.'.format(n_val, val_dataset.cardinality()))
    print('Target test size {0}, actual size {1}.'.format(n_test, test_dataset.cardinality()))

    img_size = int(train_dataset.element_spec['image'].shape[0])
    lines, angles = instantiate_radial_vectors(img_size + margin_padding, img_size + margin_padding,
                                               beam_set_size=beam_set_size,
                                               max_len=beam_length)
    train_dataset = preprocess(train_dataset, lines, angles, target_size=img_size + margin_padding,
                               batch_size=batch_size, path='./training_dataset', continuous=continuous)
    val_dataset = preprocess(val_dataset, lines, angles, target_size=img_size + margin_padding,
                             batch_size=batch_size, path='./val_dataset', continuous=continuous)
    test_dataset = preprocess(test_dataset, lines, angles, target_size=img_size + margin_padding,
                              batch_size=batch_size, path='./test_dataset', continuous=continuous)

    _, n_vec, _, n_pixels, n_channels = train_dataset.element_spec['vec'].shape

    # model and optimizer init + training start
    beams = tf.keras.layers.Input([2, n_vec, 3, n_pixels, n_channels], name='beams')
    bic = BIC(hidden=hidden_size, activation=tf.nn.leaky_relu, context=context,
              l2_regularization=0.0, edge_factor=0.5, gcn_layers=gcn_layers, dropout=0.0,
              size_vector_field=n_vec, pixel_count_per_vector=n_pixels)
    prior, unit_vec, beamencoding, ctx, similarity, \
    beamencoding_zero, beamencoding_theta, angle_energy, rnn_encoding = bic(inputs=beams)
    model = tf.keras.models.Model(inputs=beams, name='bic',
                                  outputs=(prior, unit_vec, beamencoding, ctx, similarity, \
                                           beamencoding_zero, beamencoding_theta, angle_energy, rnn_encoding))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    epochs = int(tf.math.ceil(n_iter / train_dataset.cardinality()))
    training(model, train_dataset, val_dataset, test_dataset, optimizer, lines,
             tf.cast(angles, tf.float32), epochs=epochs, name=name,
             continuous=continuous, prior=prior)

parser = argparse.ArgumentParser(description='BIC Training')
parser.add_argument('--num_beams', type=int, nargs=1, default=16, required=False,
                    help='Cardinality of the beam set (|B|): Keep the reflection symmetry assumption in mind, '
                         + 'i.e., 4, 8, 16, 32 are valid values.')
parser.add_argument('--dataset', type=str, nargs=1, default='coil100', required=False,
                    help='The dataset identifier string: fashion_mnist, cifar10, coil100, or lfw.')
parser.add_argument('--beam_length', type=int, nargs=1, default=64, required=False,
                    help='The beam length. We recommend: fashion_mnist: 14, cifar10: 16, coil100: 64, lfw: 125')
parser.add_argument('--learning_rate', type=float, nargs=1, default=0.0001, required=False,
                    help='The learning rate for the optimizer.')
parser.add_argument('--context', type=bool, nargs=1, default=True, required=False,
                    help='Whether or not to use the context node.')
parser.add_argument('--continuous', type=bool, nargs=1, default=False, required=False,
                    help='Whether or not to use continuous rotations.')
parser.add_argument('--gcn_layers', type=int, nargs=1, default=3, required=False,
                    help='Number of GCN layers.')
parser.add_argument('--batch_size', type=int, nargs=1, default=128, required=False,
                    help='The size of one batch of data for training.')
parser.add_argument('--hidden_dims', type=int, nargs=1, default=128, required=False,
                    help='The size of the latent space.')
parser.add_argument('--train_steps', type=int, nargs=1, default=4096, required=False,
                    help='The number of training steps.')
parser.add_argument('--prior', type=str, nargs=1, default='off', required=False,
                    help='Toeplitz prior: off, linear, or full.')
args = parser.parse_args()

run(n_iter=args.train_steps,
    batch_size=args.batch_size,
    beam_set_size=args.num_beams,
    name=str(args.dataset[0]),
    dataset=str(args.dataset[0]),
    splits=[0.8, 0.1, 0.1],
    partial=None,
    context=args.context,
    continuous=args.continuous,
    prior=args.prior,
    learning_rate=args.learning_rate,
    gcn_layers=args.gcn_layers,
    beam_length=args.beam_length,
    hidden_size=args.hidden_dims)
