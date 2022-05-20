import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np
from src.utils import compute_endpoints
from src.utils import instantiate_subvec_field
from src.utils import infer_angles_from_vectors
import math


def load_dataset(dataset_identifier, train_portion='75%', test_portion='25%', partial=None):
    """
    :param dataset_identifier:
    :param train_portion:
    :return: dataset with (image, label)
    """
    # splits are not always supported
    # split = ['train[:{0}]'.format(train_portion), 'test[{0}:]'.format(test_portion)]
    ds = tfds.load(dataset_identifier, split='train', shuffle_files=True)
    if partial is not None:
        ds = ds.take(partial)
    return ds


def to_dataset(original: np.array, data=None, canonic_zero=None, data_placeholder_shape=None):
    """ Dateset of tuples:
        - data: optional the vector sequences for training
        - original: the original image
        - canonic_zero: optional ground truth
    """
    if data_placeholder_shape is None and data is None:
        raise ValueError()
    elif data_placeholder_shape is not None and data is None:
        data = np.zeros(data_placeholder_shape)
    canonic_zero = canonic_zero if canonic_zero is not None else np.zeros([data.shape[0], data.shape[1]])
    dataset = tf.data.Dataset.from_tensor_slices((data, original, canonic_zero))
    return dataset


def rotate_image(image, angles: list, angle=None):
    if angle is not None:
        return {'rotated': tfa.image.rotate(image, angle, interpolation='bilinear'),
                'angle': angle}
    # Dataset does not execute eagerly, so the python randint is executed once to create the
    # graph and the returned value is used.
    rnd_idx = int(tf.random.uniform([], minval=0, maxval=len(angles)))
    # 360 - angle since `tfa.image.rotate` will rotate counter clockwise
    # but the angle matrix in the model is build for clockwise rotations
    angle = (angles[rnd_idx] / 180.) * math.pi
    # angle = 2 * math.pi - angle
    return {'rotated': tfa.image.rotate(image, angle, interpolation='bilinear'),
            'angle': tf.one_hot(rnd_idx, len(angles))}


def line_eval(image: tf.Tensor, lines: tf.Tensor):
    """
    :param image: width x height x channels
    :param lines: lines x proximity x pixels x 2
    :return:
    """
    return tf.gather_nd(image, lines)


def map_radial_vectors(dataset: tf.data.Dataset, lines: tf.Tensor, angles: list, continuous=False,
                       target_width=512, target_height=512, normalize=True,
                       horizontal_translation=0, vertical_translation=0) -> tf.data.Dataset:
    # todo if the shapes are not consistent in the ds, then the shape is None
    angles = tf.cast(angles, tf.float32)
    # normalize
    if normalize:
        label_key = 'label' if 'label' in dataset.element_spec.keys() else 'object_id'
        dataset = dataset.map(lambda x: {'image': tf.cast(x['image'], tf.float32) / 255., 'label': x[label_key]})
    # add padding
    dataset = dataset.map(lambda x: {'image': tf.pad(x['image'], [
        [tf.math.maximum(0, ((target_height - x['image'].shape[1]) // 2) - horizontal_translation),
         tf.math.maximum(0, ((target_height - x['image'].shape[1]) // 2) + horizontal_translation)],
        [tf.math.maximum(0, ((target_width - x['image'].shape[0]) // 2) - vertical_translation),
         tf.math.maximum(0, ((target_width - x['image'].shape[0]) // 2) + vertical_translation)],
        [0, 0]], "CONSTANT"), 'label': x['label']})
    # rotate images
    if continuous:
        angle = tf.random.uniform(shape=[dataset.cardinality()], minval=0, maxval=2*math.pi)
        dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices({'angle': angle})))
        dataset = dataset.map(lambda x, y: {**x, **rotate_image(x['image'], angles, angle=y['angle'])})
    else:
        dataset = dataset.map(lambda x: {**x, **rotate_image(x['image'], angles)})
    # map vector evaluations
    dataset = dataset.map(lambda x: {**x, 'beam': line_eval(x['image'], lines),
                                          'beam_rot': line_eval(x['rotated'], lines)})
    return dataset


def instantiate_radial_vectors(w_dim: int, h_dim: int, beam_set_size: int, max_len=16):
    center = (w_dim // 2, h_dim // 2)
    endpoints = compute_endpoints(w_dim, h_dim, vecs_per_quarter=(beam_set_size + 4) // 4)
    angles = infer_angles_from_vectors(endpoints, center)
    lines = instantiate_subvec_field(center, endpoints, length=max_len)
    # sanity check and validate
    lines = tf.clip_by_value(tf.cast(lines, tf.int32), 0, w_dim - 1)
    return lines, angles


def preprocess(dataset, lines, angles, target_size=32, continuous=False,
               batch_size=32, path='./training_dataset',
               horizontal_translation=0, vertical_translation=0):
    dataset = map_radial_vectors(dataset, lines, angles, normalize=True, continuous=continuous,
                                 target_width=target_size, target_height=target_size,
                                 horizontal_translation=horizontal_translation,
                                 vertical_translation=vertical_translation)
    # create generators used for training and validation
    dataset = dataset.shuffle(512).batch(batch_size).prefetch(buffer_size=100)
    if path is not None:
        tf.data.experimental.save(dataset, path)
    return dataset


def as_generator(dataset):
    while True:
        for d in dataset:
            yield d
