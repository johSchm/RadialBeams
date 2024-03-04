import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np
from src.utils import compute_endpoints
from src.utils import instantiate_beams
from src.utils import infer_angles_from_vectors
from src.utils import polar_transform, apply_circular_mask
import math


def preprocess_dataset(name, batch_size=None, image_size=None, n_beams=None, radius=None):
    def preprocess(example):
        image = tf.cast(example['image'], tf.float32) / 255.
        image = tf.image.resize(image, [image_size, image_size], antialias=True)
        image = apply_circular_mask(image)
        return {
            'image': image,
            'label': example['label'],
            'polar': polar_transform(image, radius=radius)
        }

    def preprocess_test(example):
        example = preprocess(example)
        # random rotation angle
        k = tf.random.uniform(shape=(), minval=-n_beams // 2, maxval=n_beams // 2, dtype=tf.int64)
        angle = -tf.cast(k, tf.float32) * math.pi / (n_beams // 2)
        # resample + rotation
        polar_resrot = tf.roll(example['polar'], k, axis=1)
        # rotation + resample
        polar_rotres = tfa.image.rotate(example['image'], interpolation='bilinear', angles=angle)
        polar_rotres = polar_transform(polar_rotres, radius=radius)
        return {
            'image': example['image'],
            'label': example['label'],
            'polar': example['polar'],
            'polar_resrot': polar_resrot,
            'polar_rotres': polar_rotres,
            'k': k,
            'angle': angle
        }

    if batch_size is not None and image_size is not None and radius is not None and n_beams is not None:

        train_dataset = tfds.load(name, split='train', shuffle_files=False)
        test_dataset = tfds.load(name, split='test', shuffle_files=False)

        # todo map after batch? and rewrite map for batched data to increase runtime performance
        train_dataset = (train_dataset.map(preprocess, num_parallel_calls=8).cache()
                         .batch(batch_size).prefetch(batch_size))
        test_dataset = (test_dataset.map(preprocess_test, num_parallel_calls=8).cache()
                        .batch(batch_size).prefetch(batch_size))

        train_dataset.save('./data/{}_train'.format(name))
        test_dataset.save('./data/{}_test'.format(name))

        print("Datasets saved.")

    else:

        train_dataset = tf.data.Dataset.load('./data/{}_train'.format(name)).cache().prefetch(1000)
        test_dataset = tf.data.Dataset.load('./data/{}_test'.format(name)).cache().prefetch(1000)

    return train_dataset, test_dataset


def load_dataset(dataset_identifier, train_portion='75%', test_portion='25%', partial=None):
    """
    :param dataset_identifier:
    :param train_portion:
    :return: dataset with (image, label)
    """
    # splits are not always supported
    # split = ['train[:{0}]'.format(train_portion), 'test[{0}:]'.format(test_portion)]
    try:
        ds = tfds.load(dataset_identifier, split='train', shuffle_files=True)
    except ValueError:
        ds = tfds.load(dataset_identifier, split='test', shuffle_files=True)
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


def rotate_image(image, angles: list, angle=None, fill_value=0.0):
    if angle is not None:
        return {'rotated': tfa.image.rotate(image, angle, interpolation='bilinear'),
                'angle': angle}
    # Dataset does not execute eagerly, so the python randint is executed once to create the
    # graph and the returned value is used.
    rnd_idx = int(tf.random.uniform([], minval=0, maxval=len(angles)))
    # 360 - angle since `tfa.image.rotate` will rotate counter clockwise
    # but the angle matrix in the model is build for clockwise rotations
    angle = (angles[rnd_idx] / 180.) * math.pi
    # angle = 1 * math.pi - angle
    return {'rotated': tfa.image.rotate(image, angle, interpolation='bilinear', fill_value=fill_value),
            'angle': tf.one_hot(rnd_idx, len(angles))}


def line_eval(image: tf.Tensor, lines: tf.Tensor):
    """
    :param image: width x height x channels
    :param lines: lines x proximity x pixels x 1
    :return:
    """
    return tf.gather_nd(image, lines)


def map_radial_vectors(dataset: tf.data.Dataset, lines: tf.Tensor, angles: list, continuous=False,
                       target_width=512, target_height=512, normalize=True, padding='default',
                       horizontal_translation=0, vertical_translation=0) -> tf.data.Dataset:
    # todo if the shapes are not consistent in the ds, then the shape is None
    # todo tf.tensordot(d['image'][-1, -1], tf.cast([0.2989, 0.5870, 0.1140], d['image'].dtype), axes=0)
    angles = tf.cast(angles, tf.float32)
    # normalize
    if normalize:
        label_key = 'label' if 'label' in dataset.element_spec.keys() else 'object_id'
        dataset = dataset.map(lambda x: {'image': tf.cast(x['image'], tf.float32) / 255., 'label': x[label_key]})
    # add padding
    print('Padding mode ' + padding)
    dataset = dataset.map(lambda x: {'image': tf.pad(x['image'], [
        [tf.math.maximum(0, ((target_height - x['image'].shape[1]) // 2) - horizontal_translation),
         tf.math.maximum(0, ((target_height - x['image'].shape[1]) // 2) + horizontal_translation)],
        [tf.math.maximum(0, ((target_width - x['image'].shape[0]) // 2) - vertical_translation),
         tf.math.maximum(0, ((target_width - x['image'].shape[0]) // 2) + vertical_translation)],
        [0, 0]], "CONSTANT",
        constant_values=x['image'][-1, -1, 0] if padding == 'adaptive' else 0.0),
        'label': x['label']})
    # rotate images
    if continuous:
        angle = tf.random.uniform(shape=[dataset.cardinality()], minval=0, maxval=2*math.pi)
        dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices({'angle': angle})))
        dataset = dataset.map(lambda x, y: {**x, **rotate_image(x['image'], angles, angle=y['angle'],
                              fill_value=x['image'][-1, -1, 0] if padding == 'adaptive' else 0.0)})
    else:
        dataset = dataset.map(lambda x: {**x, **rotate_image(x['image'], angles,
                              fill_value=x['image'][-1, -1, 0] if padding == 'adaptive' else 0.0)})
    # map vector evaluations
    dataset = dataset.map(lambda x: {**x, 'beam': line_eval(x['image'], lines),
                                          'beam_rot': line_eval(x['rotated'], lines)})
    return dataset


def instantiate_radial_beams(w_dim: int, h_dim: int, beam_set_size: int, max_len=16):
    center = (w_dim // 2, h_dim // 2)
    endpoints = compute_endpoints(w_dim, h_dim, beams_per_quarter=(beam_set_size + 4) // 4)
    angles = infer_angles_from_vectors(endpoints, center)
    lines = instantiate_beams(center, endpoints, length=max_len, w_dim=w_dim)
    # sanity check and validate
    lines = tf.clip_by_value(tf.cast(lines, tf.int32), 0, w_dim - 1)
    return lines, angles


def preprocess(dataset, lines, angles, target_size=32, continuous=False,
               batch_size=32, path='./training_dataset', padding='default',
               horizontal_translation=0, vertical_translation=0):
    dataset = map_radial_vectors(dataset, lines, angles, normalize=True, continuous=continuous,
                                 target_width=target_size, target_height=target_size,
                                 horizontal_translation=horizontal_translation,
                                 vertical_translation=vertical_translation,
                                 padding=padding)
    # create generators used for training and validation
    dataset = dataset.shuffle(512).batch(batch_size).prefetch(buffer_size=100)
    if path is not None:
        tf.data.experimental.save(dataset, path)
    return dataset


def as_generator(dataset):
    while True:
        for d in dataset:
            yield d
