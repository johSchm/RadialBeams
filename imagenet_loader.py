import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from src.utils import apply_circular_mask, polar_transform
from datasets import load_from_disk

print(os.getcwd())
dataset = load_from_disk('../generatornet/imagenet-1k').train_test_split(test_size=0.2)

train_dataset = dataset['train']
test_dataset = dataset['test']

center_cropping = tf.keras.layers.CenterCrop(224, 224)
# @tf.function
def preprocess(example):
    image = example['image'].convert('RGB')
    image = tf.cast(image, tf.float32) / 255.
    image = center_cropping(image)
    image = apply_circular_mask(image)
    return {
        'image': image,
        'label': example['label'],
        'polar': polar_transform(image)
    }


# train_dataset = train_dataset.map(preprocess)
# train_dataset.save_to_disk('imagenet-train-polar.hf')

test_dataset = test_dataset.map(preprocess)
test_dataset.save_to_disk('imagenet-test-polar.hf')
