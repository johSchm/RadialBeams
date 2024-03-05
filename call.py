import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='BIC Training')
parser.add_argument('--gpu',
                    default=0, type=int, required=False,
                    help='The GPU number.')
parser.add_argument('--name',
                    default=datetime.now().strftime("%m%d-%H%M"), type=str, required=False,
                    help='Logging name for weights and biases.')
parser.add_argument('--dataset',
                    default='stanford_dogs', type=str, required=False,
                    choices=('stanford_dogs', ),
                    help='Training and Testing dataset.')
parser.add_argument('--model',
                    default='PolarRegressor2D', type=str, required=False,
                    choices=('PolarRegressor1D', 'PolarRegressor2D'),
                    help='The model used during training and inference.')
parser.add_argument('--noise',
                    default=0.1, type=float, required=False,
                    help='Gaussian Additive Noise during training [0,1].')
parser.add_argument('--batch_size',
                    default=200, type=int, required=False,
                    help='Batch Size for training and testing.')
parser.add_argument('--n_filters',
                    default=64, type=int, required=False,
                    help='Number of conv filters.')
parser.add_argument('--n_epochs',
                    default=250, type=int, required=False,
                    help='Number of Training epochs.')
parser.add_argument('--learning_rate',
                    default=5e-4, type=float, required=False,
                    help='The learning rate for training.')
parser.add_argument('--recreate_datasets',
                    default=False, type=bool, required=False,
                    help='This will load the base dataset and pre-process it again.')
args = parser.parse_args()
print(args)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import math
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from src.learning import get_laplace
from src.utils import (log_biases, log_weights, log_gradients)
from src.visu import (plot_conv_filters, saliency_map, weighted_saliency_map, grad_cam, energy_map,
                      plot_output_shift, process_until)
from src.parsing import preprocess_dataset
from src.model import PolarRegressor1D, PolarRegressor2D


if args.dataset == 'stanford_dogs':
    image_size = 100
else:
    raise ValueError('')

if args.dataset == 'stanford_dogs':
    n_channels = 3
else:
    n_channels = 1

# todo add args option? -> note, 3 is enough but this requires altering the architecture
radius = image_size//2 - 3

config = {
    "dataset": args.dataset,
    "image_size": image_size,
    "noise_factor": args.noise,
    "batch_size": args.batch_size,
    "n_filters": args.n_filters,
    "n_epochs": args.n_epochs,
    "n_channels": n_channels,
    "model": args.model,
    "learning_rate": args.learning_rate,
    "radius": radius,
    "len_beam": int(round(radius)),
    "n_beams": int(round(radius*2*math.pi))
}
wandb.init(project="RadialBeams", config=config, group=config['dataset'], name=args.name)

# dataset loading
# todo add notification if batch_size does not match loaded batch size
if args.recreate_datasets:
    train_dataset, test_dataset = preprocess_dataset(args.dataset, args.batch_size, image_size,
                                                     config["n_beams"], config["radius"])
else:
    train_dataset, test_dataset = preprocess_dataset(args.dataset)
print('Loaded dataset with sample shape {}.'.format(next(iter(test_dataset))['polar'].shape))

# model compilation
if args.model == 'PolarRegressor1D':
    model = PolarRegressor1D(len_beam=config['len_beam'], n_beams=config['n_beams'],
                             n_channels=config['n_channels'], n_filters=config['n_filters'])
elif args.model == 'PolarRegressor2D':
    model = PolarRegressor2D(len_beam=config['len_beam'], n_beams=config['n_beams'],
                             n_channels=config['n_channels'], n_filters=config['n_filters'])
else:
    raise ValueError('Unrecognized model: {}'.format(args.model))
model.build(input_shape=(config['batch_size'], config['len_beam'], config['n_beams'], config['n_channels']))
model(tf.zeros((config['batch_size'], config['len_beam'], config['n_beams'], config['n_channels'])))
model.summary()

wandb.log({'Number of (learnable) Parameters': np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])})

# optimiser and label
lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(args.learning_rate, 1000)
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_decayed_fn)#config['learning_rate'])#, weight_decay=0.005)
label = get_laplace(n_beams=config['n_beams'])

# train and test
for e in tqdm(range(config['n_epochs'])):

    # training loop
    for s, sample in enumerate(train_dataset):
        # +1 as wandb starts with 1
        step = e * len(train_dataset) + s
        image = sample["polar"] + config['noise_factor'] * tf.random.normal(sample["polar"].shape)
        # image = (image / tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

        with tf.GradientTape() as tape:
            k_distribution = model(image, training=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=k_distribution, labels=label)
            grad = tape.gradient(loss, model.trainable_variables)
        # grad = [(tf.clip_by_value(g, -1., 1.)) for g in grad]
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        wandb.log({"training loss": np.mean(loss.numpy())}, step=step)
        wandb.log({"learning rate": optimizer.lr.numpy()}, step=step)

        # logging
        # if s == 1 and e % 5 == 0:
        #     log_biases(model, wandb.log, step=step)
        #     log_weights(model, wandb.log, step=step)
        #     log_gradients(model, grad, wandb.log, step=step)
        #     padded_input = model.cyclic_beam_padding(image)
        #     first_layer_filters = \
        #     model.get_layer('enc0').get_layer('rb0').get_layer('main').get_layer('cv1').get_weights()[0]
        #     first_resid_filters = \
        #     model.get_layer('enc0').get_layer('rb0').get_layer('res').get_layer('cv').get_weights()[0]
        #     wandb.log({"Conv Filters (first main)": plot_conv_filters(first_layer_filters)}, step=step)
        #     wandb.log({"Conv Filters (first res)": plot_conv_filters(first_resid_filters)}, step=step)
        #     wandb.log({"Saliency Map": saliency_map(image, model, label)}, step=step)
        #     wandb.log({"Weighted Saliency Map": weighted_saliency_map(image, model, label)}, step=step)
        #     wandb.log({"GradCAM": grad_cam(image, model)}, step=step)
        #     wandb.log({"Energy Map": energy_map(image, model)}, step=step)
        #     wandb.log({"Distribution Shift": plot_output_shift(image, model, normalise=True)}, step=step)
        #     feature_map = process_until(model.get_layer('enc0').get_layer('rb0').get_layer('main').input,
        #                                 model.get_layer('enc0').get_layer('rb0').get_layer('main').get_layer(
        #                                     'cv1').output, padded_input)
        #     wandb.log({"Feature Map (first main)": feature_map}, step=step)
        #     feature_map = process_until(model.get_layer('enc0').get_layer('rb0').get_layer('res').input,
        #                                 model.get_layer('enc0').get_layer('rb0').get_layer('res').get_layer(
        #                                     'cv').output, padded_input)
        #     wandb.log({"Feature Map (first res)": feature_map}, step=step)
        #     wandb.log({"Padded Input": plt.imshow(padded_input[0])}, step=step)
        #     plt.close('all')

        # equivariance test loop
        # image, k = random_roll(image)
        # test_loss = tf.abs(k + config['n_beams']//2-tf.argmax(model(image), axis=-1))
        # wandb.log({"equivariance test": np.mean(test_loss.numpy())}, step=step)

    # pseudo-equivariant test loop
    test_resrot, test_rotres = [], []
    for s, sample in enumerate(test_dataset):
        test_resrot.append(tf.abs(sample['k'] + config['n_beams'] // 2
                                  - tf.argmax(model(sample['polar_resrot']), axis=-1)).numpy())
        test_rotres.append(tf.abs(sample['k'] + config['n_beams'] // 2
                                  - tf.argmax(model(sample['polar_rotres']), axis=-1)).numpy())

    wandb.log({"ResRot Test": np.concatenate(test_resrot, axis=0).mean()}, step=step)
    wandb.log({"RotRes Test": np.concatenate(test_rotres, axis=0).mean()}, step=step)

    # store model weights
    if e % 5 == 0:
        name = './model/{0}_{1}.tf'.format(config['dataset'], e)
        model.save(name)
        wandb.save(name)
