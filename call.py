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
from src.parsing import instantiate_radial_beams
import argparse


def run(n_iter=4*1024, batch_size=1*128, beam_set_size=8, name='', context=True, prior='off',
        dataset_name='coil100', splits=[0.8, 0.1, 0.1], partial=None, continuous=False,
        learning_rate=0.0001, gcn_layers=3, beam_length=64, hidden_size=128):

    dataset = load_dataset(dataset_name, partial=partial)
    img_size = dataset.element_spec['image'].shape[0]

    if dataset_name == 'mnist':
        # remove 9 from MNIST since equal 6 if rotated
        # filter function of tf.data does not seem to work
        images, labels, = [], []
        for d, data in enumerate(dataset):
            print('Filtering MNIST {0}/{1}'.format(d, dataset.cardinality() - 1), end='\r')
            if data['label'] != 9:
                images.append(data['image'])
                labels.append(data['label'])
        dataset = tf.data.Dataset.from_tensor_slices({'image': images, 'label': labels})

    margin_padding = math.ceil(img_size * (math.sqrt(2) - 1))

    n_train = int(splits[0] * float(dataset.cardinality()))
    n_val = int(splits[1] * float(dataset.cardinality()))
    n_test = int(splits[2] * float(dataset.cardinality()))

    train_dataset = dataset.take(n_train)
    val_dataset = dataset.skip(n_train).take(n_val)
    test_dataset = dataset.skip(n_train).skip(n_val).take(n_train)

    print('Target train size {0}, actual size {1}.'.format(n_train, train_dataset.cardinality()))
    print('Target val size {0}, actual size {1}.'.format(n_val, val_dataset.cardinality()))
    print('Target test size {0}, actual size {1}.'.format(n_test, test_dataset.cardinality()))

    lines, angles = instantiate_radial_beams(img_size + margin_padding, img_size + margin_padding,
                                             beam_set_size=beam_set_size,
                                             max_len=beam_length)
    train_dataset = preprocess(train_dataset, lines, angles, target_size=img_size + margin_padding,
                               batch_size=batch_size, path='./training_dataset', continuous=continuous,
                               padding='adaptive' if dataset_name == 'coil100' else 'default')
    val_dataset = preprocess(val_dataset, lines, angles, target_size=img_size + margin_padding,
                             batch_size=batch_size, path='./val_dataset', continuous=continuous,
                             padding='adaptive' if dataset_name == 'coil100' else 'default')
    test_dataset = preprocess(test_dataset, lines, angles, target_size=img_size + margin_padding,
                              batch_size=batch_size, path='./test_dataset', continuous=continuous,
                              padding='adaptive' if dataset_name == 'coil100' else 'default')

    _, n_beams, _, n_pixels, n_channels = train_dataset.element_spec['beam'].shape

    # model and optimizer init + training start
    beams = tf.keras.layers.Input([2, n_beams, 3, n_pixels, n_channels], name='beams')
    bic = BIC(hidden=hidden_size, activation=tf.nn.leaky_relu, context=context,
              l2_regularization=0.0, edge_factor=0.5, gcn_layers=gcn_layers, dropout=0.2,
              n_beams=n_beams, pixel_count_per_beam=n_pixels, lstm_layers=3)
    _ = bic(inputs=beams)
    bic.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    epochs = int(tf.math.ceil(n_iter / train_dataset.cardinality()))
    training(bic, train_dataset, val_dataset, test_dataset, optimizer, lines,
             tf.cast(angles, tf.float32), epochs=epochs, name=name,
             continuous=continuous, prior=prior)

parser = argparse.ArgumentParser(description='BIC Training')
parser.add_argument('--num_beams', type=int, nargs=1, default=16, required=False,
                    help='Cardinality of the beam set (|B|): Keep the reflection symmetry assumption in mind, '
                         + 'i.e., 4, 8, 16, 32 are valid values.')
parser.add_argument('--dataset', type=str, nargs=1, default='mnist', required=False,
                    help='The dataset identifier string: mnist, fashion_mnist, cifar10, coil100, or lfw.')
parser.add_argument('--beam_length', type=int, nargs=1, default=14, required=False,
                    help='The beam length. We recommend: fashion_mnist: 14, cifar10: 16, coil100: 64, lfw: 125')
parser.add_argument('--learning_rate', type=float, nargs=1, default=0.01, required=False,
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
parser.add_argument('--train_steps', type=int, nargs=1, default=2048, required=False,
                    help='The number of training steps.')
parser.add_argument('--prior', type=str, nargs=1, default='off', required=False,
                    help='Toeplitz prior: off, only, linear or equal.')
args = parser.parse_args()

run(n_iter=args.train_steps,
    batch_size=args.batch_size,
    beam_set_size=args.num_beams,
    name=str(args.dataset) + '_' + str(args.prior) + '_' + str(args.num_beams) + '_' + str(args.continuous),
    dataset_name=str(args.dataset),
    splits=[0.8, 0.1, 0.1],
    partial=None,
    context=args.context,
    continuous=args.continuous,
    prior=args.prior,
    learning_rate=args.learning_rate,
    gcn_layers=args.gcn_layers,
    beam_length=args.beam_length,
    hidden_size=args.hidden_dims)
