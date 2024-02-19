import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
#
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# from src.utils import apply_circular_mask, polar_transform
from datasets import load_from_disk
import numpy as np

train_dataset = load_from_disk('./data/imagenet-train-circular-224-v2.hf')#.train_test_split(test_size=0.2)
test_dataset = load_from_disk('./data/imagenet-test-circular-224-v2.hf')#.train_test_split(test_size=0.2)

# train_dataset = dataset['train']
# test_dataset = dataset['test']

# center_cropping = tf.keras.layers.CenterCrop(224, 224)

def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]

def create_circular_mask(height, width, center=None, radius=None):
    if center is None:
        center = [height // 2, width // 2]
    if radius is None:
        radius = min(center[0], center[1], height - center[0], width - center[1])

    y, x = np.ogrid[:height, :width]
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    return mask.astype(np.float32)

def apply_circular_mask(image):
    """
    Args:
        image: (height x width x channel)
    """
    height, width, _ = image.shape

    center = [height // 2, width // 2]
    radius = min(center[0], center[1])

    mask = create_circular_mask(height, width, center, radius)

    return image * mask[..., None]

def preprocess(example):
    image = example['image'].convert('RGB')
    image = np.asarray(image, dtype=np.float32)
    # image = tf.cast(image, tf.float32) / 255.
    if image.shape[0] > 224:
        crop_start = image.shape[0] // 2 - 224 // 2
        image = image[crop_start:crop_start + 224]
    if image.shape[1] > 224:
        crop_start = image.shape[1] // 2 - 224 // 2
        image = image[:, crop_start:crop_start + 224]
    image = apply_circular_mask(image)
    return {
        'image': image / 255.,
        'label': example['label'],
        # 'polar': polar_transform(image)
    }

def has_min_size(example):
    im = np.asarray(example['image'], dtype=np.float32)
    return im.shape[0] >= 224 and im.shape[1] >= 224


def batcher(batch):
    return {"image": np.array(batch['image']), "label": np.array(batch['label'])}


# train_dataset = train_dataset.filter(has_min_size, num_proc=64)
# train_dataset = train_dataset.map(preprocess, batched=False, batch_size=512, num_proc=64)
# train_dataset.save_to_disk('./data/imagenet-train-circular-224-v2.hf')
# train_dataset.map(batcher, batched=True, batch_size=48, num_proc=64)
# train_dataset.save_to_disk('./data/imagenet-train-circular-batched-48.hf')
train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset["image"], train_dataset["label"]))
tf.data.experimental.save(train_dataset, './data/imagenet-train-circular')

# test_dataset = test_dataset.filter(has_min_size, num_proc=64)
# test_dataset = test_dataset.map(preprocess, batched=False, batch_size=512, num_proc=64)
# test_dataset.save_to_disk('./data/imagenet-test-circular-224-v2.hf')
# test_dataset = test_dataset.map(batcher, batched=True, batch_size=48, num_proc=64)
# test_dataset.save_to_disk('./data/imagenet-test-circular-batched-48-v2.hf')
