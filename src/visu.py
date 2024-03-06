import copy
import matplotlib.pyplot as plt
import numpy as np
# import imutils
# import pandas as pd
# import seaborn as sns
# from sklearn.decomposition import PCA
from PIL import Image, ImageDraw
# from sklearn.manifold import TSNE
import tensorflow as tf
import math
import tensorflow_addons as tfa
# import wandb
from matplotlib.patches import Rectangle
from src.utils import polar_transform_inv
from matplotlib.colors import Normalize


def plot_in_polar_coors(image, ax=None):
    # Convert RGB image to grayscale
    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Define polar coordinates
    theta = np.linspace(0, 2 * np.pi, grayscale_image.shape[1])
    r = np.linspace(0, 1, grayscale_image.shape[0])
    Theta, R = np.meshgrid(theta, r)

    # Map grayscale intensity to color
    norm = Normalize(vmin=0, vmax=255)  # Assuming 8-bit grayscale (0-255)
    colors = plt.cm.gray(norm(grayscale_image))

    # Create the polar plot
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.pcolormesh(Theta, R, grayscale_image, shading='auto', cmap='gray')

    # Customize the plot (add labels, title, etc.)
    ax.set_rticks([])  # Hide radial ticks for cleaner appearance
    return ax


def polar_energy_plot(energies, polar_image, ax=None, bottom=1, max_height=1):
    N = energies.shape[0]
    energies = (energies - np.min(energies)) / (np.max(energies) - np.min(energies))
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = (2 * np.pi) / N

    if ax is None:
        ax = plt.subplot(111, polar=True)

    plot_in_polar_coors(polar_image, ax=ax)
    bars = ax.bar(theta, energies, width=width, bottom=bottom)

    # Use custom colors and opacity
    for e, bar in zip(energies, bars):
        bar.set_facecolor(plt.cm.Blues(e))
        bar.set_alpha(0.8)

    # this is the offset added by the polar transform function (0. degree at [0,1])
    ax.set_theta_offset(-np.pi / 2)

    # ax.set_theta_zero_location('N') # set zero to north
    ax.axis("off")
    return ax


def plot_pred_polar_grid(model, sample, n_samples=4):
    plt.ioff()
    fig, axs = plt.subplots(3, n_samples, subplot_kw={'projection': 'polar'}, figsize=(n_samples * 2, 5))
    for k, key in enumerate(['polar', 'polar_resrot', 'polar_rotres']):
        polar_images = sample[key][:n_samples]
        n_beams = polar_images[0].shape[1]
        preds = model(polar_images, training=False, ema=True)[:n_samples]
        pred_angle = (tf.cast(tf.argmax(preds, axis=-1) - n_beams // 2, float) * 2 * math.pi) / n_beams
        for i, (image, pred) in enumerate(zip(polar_images, preds)):
            polar_energy_plot(pred, image, ax=axs[k, i])
            if key == 'polar':
                axs[k, i].vlines(math.pi, 1., 2., colors='orange', zorder=100, linestyles='dotted')
            else:
                axs[k, i].vlines(-sample['angle'][i] + math.pi, 1., 2., colors='orange', zorder=100,
                                 linestyles='dotted')
            axs[k, i].vlines(pred_angle[i] + math.pi, 1., 2., colors='orange', zorder=100, linestyles='solid')

    axs[0, 0].annotate('Original', (4.8, 3), fontsize=6, va='center', annotation_clip=False, rotation=90)
    axs[1, 0].annotate('Resampling + Rotation', (4.8, 3), fontsize=6, va='center', annotation_clip=False, rotation=90)
    axs[2, 0].annotate('Rotation + Resampling', (4.8, 3), fontsize=6, va='center', annotation_clip=False, rotation=90)

    plt.savefig('test_samples.png')
    # return Image.open('test_samples.pdf')


def polar_pred_plot(model, polar, normalise=False):
    pred = np.exp(model(polar).numpy()[0])
    if normalise:
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = plt.cm.hot(pred)
    pred = np.tile(pred[None], (10, 1, 1))[..., :3]
    return np.concatenate([polar[0].numpy(), pred], axis=0), pred

def plot_output_shift(image, model, normalise=False):
    fig, axs = plt.subplots(2,1, figsize=(8,3))
    img1, pred1 = polar_pred_plot(model, image, normalise=normalise)
    img2, pred2 = polar_pred_plot(model, tf.roll(image, 50, axis=2), normalise=normalise)
    im = axs[0].imshow(img1)
    axs[1].imshow(img2)
    axs[0].axis('off')
    axs[1].axis('off')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=Normalize(vmin=pred1.min(), vmax=pred1.max()))
    plt.colorbar(sm, cax=fig.add_axes([0.87, 0.2, 0.02, 0.6]), orientation='vertical')
    # plt.tight_layout()
    return fig

def saliency_map(image, model, label):
    image = tf.stack([image[0], tf.roll(image[0], 100, axis=1)], axis=0)
    # k = tf.one_hot(tf.range(n_beams), n_beams, dtype=tf.float32)
    # k = tf.transpose(k, (1, 0))
    with tf.GradientTape() as tape2:
        tape2.watch(image)
        # diag part as [batch x n_beams] @ [n_beams x n_beams(range)] = [batch x n_beams]
        # loss = tf.linalg.diag_part(1. - (model(image) @ k))
        # loss = 1. - (model(image) @ get_gt_beam_energy(epoch)[:, None])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=model(image), labels=label)
        # [batch x beam_len x n_beams x channels(rgb)]
        grad = tape2.gradient(loss, image)
    # max-out channels -> [batch x beam_len x n_beams]
    dgrad_max_ = np.max(tf.math.abs(grad), axis=-1)
    arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(polar_transform_inv(image[0]), alpha=0.5)
    axs[0].imshow(plt.cm.hot(polar_transform_inv(grad_eval[0, ..., None])[..., 0]), alpha=0.5)
    axs[1].imshow(polar_transform_inv(image[1]), alpha=0.5)
    axs[1].imshow(plt.cm.hot(polar_transform_inv(grad_eval[1, ..., None])[..., 0]), alpha=0.5)
    axs[0].axis("off")
    axs[1].axis("off")
    plt.tight_layout()
    return fig


def weighted_saliency_map(image, model, label):
    image = tf.stack([image[0], tf.roll(image[0], 100, axis=1)], axis=0)
    # k = tf.one_hot(tf.range(n_beams), n_beams, dtype=tf.float32)
    # k = tf.transpose(k, (1, 0))
    with tf.GradientTape() as tape2:
        tape2.watch(image)
        # diag part as [batch x n_beams] @ [n_beams x n_beams(range)] = [batch x n_beams]
        # loss = tf.linalg.diag_part(1. - (model(image) @ k))
        # loss = 1. - (model(image) @ get_gt_beam_energy(epoch)[:, None])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=model(image), labels=label)
        # [batch x beam_len x n_beams x channels(rgb)]
        grad = tape2.gradient(loss, image)
    # max-out channels -> [batch x beam_len x n_beams]
    dgrad_max_ = np.max(tf.math.abs(grad), axis=-1)
    arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(polar_transform_inv(image[0] * grad_eval[0, ..., None]))
    axs[1].imshow(polar_transform_inv(image[1] * grad_eval[1, ..., None]))
    axs[0].axis("off")
    axs[1].axis("off")
    plt.tight_layout()
    return fig


def grad_cam(image, model):
    image = tf.stack([image[0], tf.roll(image[0], 100, axis=1)], axis=0)
    with tf.GradientTape() as tape3:
        tape3.watch(image)
        preds = model(image)
        # class_channel = tf.gather(preds, tf.argmax(preds, axis=-1))
        class_channel = preds[:, tf.argmax(preds[0])]
        conv_features = model.latent_polar_map
    grads = tape3.gradient(class_channel, conv_features)
    # average pooling over batch and channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_features @ pooled_grads[..., None])
    heatmap = tf.image.resize(heatmap[..., None], (image.shape[1], image.shape[2]))
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(polar_transform_inv(image[0]), alpha=0.5)
    axs[0].imshow(plt.cm.hot(polar_transform_inv(heatmap[0])[..., 0]), alpha=0.5)
    axs[1].imshow(polar_transform_inv(image[1]), alpha=0.5)
    axs[1].imshow(plt.cm.hot(polar_transform_inv(heatmap[1])[..., 0]), alpha=0.5)
    axs[0].axis("off")
    axs[1].axis("off")
    plt.tight_layout()
    return fig

def energy_map(image, model):
    image = tf.stack([image[0], tf.roll(image[0], 100, axis=1)], axis=0)
    _ = model(image)
    conv_features = model.latent_polar_map
    conv_features = tf.reduce_max(conv_features, axis=-1)
    conv_features = (conv_features - tf.reduce_min(conv_features)) / (
                tf.reduce_max(conv_features) - tf.reduce_min(conv_features))
    conv_features = tf.image.resize(conv_features[..., None], (image.shape[1], image.shape[2]))
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(polar_transform_inv(image[0]), alpha=0.5)
    axs[0].imshow(plt.cm.hot(polar_transform_inv(conv_features[0])[..., 0]), alpha=0.5)
    axs[1].imshow(polar_transform_inv(image[1]), alpha=0.5)
    axs[1].imshow(plt.cm.hot(polar_transform_inv(conv_features[1])[..., 0]), alpha=0.5)
    axs[0].axis("off")
    axs[1].axis("off")
    plt.tight_layout()
    return fig

def plot_conv_filters(filters):
    """
    Args:
        filters: (kernel x kernel x in_features x out_features)

    Returns:
        figure
    """
    in_features, out_features = filters.shape[2:]
    fig, axs = plt.subplots(in_features, out_features, figsize=(out_features, in_features))
    for i in range(in_features):
        for o in range(out_features):
            axs[i, o].imshow(plt.cm.coolwarm(filters[..., i, o]))
            axs[i, o].axis('off')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=Normalize(vmin=filters.min(), vmax=filters.max()))
    plt.colorbar(sm, cax=fig.add_axes([0.92, 0.2, 0.01, 0.6]), orientation='vertical')
    # plt.tight_layout()
    return fig

def process_until(input_layer, output_layer, input):
    intermediate_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    output = intermediate_model(input)
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(polar_transform_inv(output.numpy().min(axis=-1)[0, ..., None]), cmap=plt.cm.hot)
    axs[1].imshow(polar_transform_inv(output.numpy().max(axis=-1)[0, ..., None]), cmap=plt.cm.hot)
    axs[2].imshow(polar_transform_inv(output.numpy().mean(axis=-1)[0, ..., None]), cmap=plt.cm.hot)
    axs[0].set_title('min')
    axs[1].set_title('max')
    axs[2].set_title('mean')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    return fig

def create_gradient_image(width, height, start_color, end_color):
  """
  Creates a two-color gradient image.

  Args:
    width: The width of the image.
    height: The height of the image.
    start_color: A list of 3 elements representing the starting color (RGB).
    end_color: A list of 3 elements representing the ending color (RGB).

  Returns:
    A NumPy array representing the gradient image (width x height x 3).
  """
  red_gradient = np.linspace(start_color[0], end_color[0], height)
  green_gradient = np.linspace(start_color[1], end_color[1], height)
  blue_gradient = np.linspace(start_color[2], end_color[2], height)
  image = np.stack([red_gradient, green_gradient, blue_gradient], axis=-1)
  image = np.tile(image[:, None], (1, width, 1))
  return image.astype(np.uint8)

def add_grad_coloring(image):
    if tf.is_tensor(image):
        image = image.numpy()
    width, height, channels = image.shape
    grad_overlay = create_gradient_image(height, width, [0,0,255], [255,0,0])
    return (image * grad_overlay) / 255.

def plot_image(image, ax, title=''):
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)


def line_overlay(lines, image, ax, title='', cmap='Greys'):
    for line in lines:
        ax.plot([line[1, 0, 0], line[1, -1, 0]], [line[1, 0, 1], line[1, -1, 1]],
                color="white", linewidth=2)
        ax.plot([line[0, 0, 0], line[0, -1, 0]], [line[0, 0, 1], line[0, -1, 1]],
                color="white", linewidth=1, linestyle='dashed')
        ax.plot([line[2, 0, 0], line[2, -1, 0]], [line[2, 0, 1], line[2, -1, 1]],
                color="white", linewidth=1, linestyle='dashed')
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)


def beams2img(lines, beam, shape, markers=False, ax=None, cmap='Greys', wandb_img=False):
    """
    :param lines: beams x proximity x beam_length x (x,y)
    :param beam: beams x proximity x beam_length x (color)
    :param shape: width x height x (color)
    :param markers:
    :param ax:
    :param cmap:
    :param wandb_img:
    :return:
    """
    image = np.zeros(shape, dtype=float)
    for line, eval_beam in zip(lines, beam):
        image[line[1, :, 0], line[1, :, 1]] = eval_beam[1]
        image[line[0, :, 0], line[0, :, 1]] = eval_beam[0]
        image[line[2, :, 0], line[2, :, 1]] = eval_beam[2]
    if markers:
        if ax is not None:
            ax.imshow(image, cmap=cmap)
            ax.scatter(lines[:, 0, :, 0], lines[:, 0, :, 1], s=20, c='red', marker='x', clip_on=False)
        plt.imshow(image, cmap=cmap)
        plt.scatter(lines[:, 0, :, 0], lines[:, 0, :, 1], s=20, c='red', marker='x', clip_on=False)
        if wandb_img:
            img = wandb.Image(plt)
            plt.close()
            return img
        return plt
    if wandb_img:
        return wandb.Image(image)
    return image


def beam_eval(image, lines, ax, beam=None, distribution=None, title='') -> np.array:
    # beam evaluations
    if beam is not None:
        image = beams2img(lines, beam, image.shape)
    # highlight selected beam
    if distribution is not None:
        highlight_index = len(distribution) - 1 - np.argmax(distribution)
        for point in lines[highlight_index, 1]:
            ax.scatter(point[0], point[1], s=20, c='red', marker='x', clip_on=False)
    ax.imshow(image)
    ax.set_title(title)
    return image


def stacked_eval(beam, ax, title=''):
    # stacked evaluations
    # beam (beams x proximity x pixels x channel)
    ax.imshow(tf.reshape(beam, [-1, tf.shape(beam)[-2], tf.shape(beam)[-1]]))
    ax.set_title(title)


def plot_rot(image, distribution, angles, ax):
    # plot the output image after rotation
    angle = 360 - angles[np.argmax(distribution)]
    rot_image = imutils.rotate(image, angle=angle.astype(int))
    ax.imshow(rot_image)
    ax.set_title('Rotated Image by %.1f degree' % angle)


def plot_hist(angles, distribution, gt_angle, ax, title=''):
    # plot the distribution of C0
    labels = []
    for angle in angles:
        if gt_angle is not None and angle == gt_angle:
            labels.append("--> %.1f" % angle)
            continue
        labels.append("%.1f" % angle)
    ax.barh(np.arange(distribution.shape[-1]), distribution,
            align='center', height=0.5, tick_label=labels)
    ax.set_title(title)


def plot_eval_beams(image, lines, beam, name: str):
    """ This will plot the evaluated beams.
    The lines contain the positional information and the beam field the evaluations.
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(name, fontsize=16)
    plot_image(image, ax=ax[0, 0], title='Raw Image')
    line_overlay(lines, image, ax=ax[0, 1], title='Beam Overlay')
    beam_eval(image, lines, beam=beam, ax=ax[1, 0], title='Beam Evaluations')
    stacked_eval(beam, ax[1, 1], title='Stacked Evaluations')
    plt.savefig('./data/' + name.replace(' ', '_') + '.pdf', dpi=300)
    plt.close()


def plot_distribution(image, beam, beam_rot, distribution, lines, angles, name: str, gt_angle=None):
    """ plot the distribution of both examples and highlight the max selected beam
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(name, fontsize=16)
    beam_eval(image, lines, ax=ax[0], beam=beam, distribution=np.eye(len(angles))[0],
             title='Original beamtor Evaluations')
    beam_eval(image, lines, ax=ax[1], beam=beam_rot, distribution=distribution,
             title='Rotated Beam Evaluations')
    plot_hist(angles, distribution, gt_angle, ax=ax[2], title='Original Prediction Histogram')
    # plot_hist(angles, distribution, gt_angle, ax=ax[0, 0], title='Original Prediction Histogram')
    plt.savefig('./data/' + name.replace(' ', '_') + '.pdf', dpi=300)
    plt.close()


def proof_of_concept(image, rot_image, distribution, lines, angles, name='', gt_angle=None):
    """ rotate the image back by the predicted angle.
    This is the deployment test phase, where only one image is feed in.
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(name, fontsize=16)
    plot_image(image, ax=ax[0, 0], title='Ground Truth Image')
    beam_eval(rot_image, lines, distribution=distribution, ax=ax[1, 0], title='Input Image w/ predicted C0')
    plot_rot(rot_image, distribution, angles, ax=ax[0, 1])
    plot_hist(360 - angles, distribution, 360 - gt_angle, ax=ax[1, 1], title='Prediction Histogram')
    plt.savefig('./data/' + name.replace(' ', '_') + '.pdf', dpi=300)
    plt.close()


def plot_training_performance(train_loss, val_loss,
                              samples_per_bucket: int, n_iterations: int, path='./training'):
    """ Plotting for training/validation curves.
    training_losses, validation_losses = {'cz_loss': [], 'a_loss': [], 'l2_loss': []}, {'cz_loss': []}
    """
    plt.close()

    iteration_bins = np.repeat(np.arange(n_iterations/samples_per_bucket),
                               samples_per_bucket) * samples_per_bucket

    # saliency check
    train_loss = {k: tf.cast(t, tf.float32)[:iteration_bins.shape[0]]
                  for k, t in zip(train_loss.keys(), train_loss.values())}
    val_loss = {k: tf.cast(t, tf.float32)[:iteration_bins.shape[0]]
                for k, t in zip(val_loss.keys(), val_loss.values())}

    # plot losses
    train_cz_loss = np.concatenate([iteration_bins[None, ...], train_loss['unit_circle_loss'][None, ...]], axis=0).transpose()
    train_a_loss = np.concatenate([iteration_bins[None, ...], train_loss['toeplitz_loss'][None, ...]], axis=0).transpose()
    train_l2_loss = np.concatenate([iteration_bins[None, ...], train_loss['l2_loss'][None, ...]], axis=0).transpose()
    val_cz_loss = np.concatenate([iteration_bins[None, ...], val_loss['unit_circle_loss'][None, ...]], axis=0).transpose()

    train_cz_loss_df = pd.DataFrame(train_cz_loss, columns=['iteration', 'train_unit_circle_loss'])
    train_a_loss_df = pd.DataFrame(train_a_loss, columns=['iteration', 'train_toeplitz_loss'])
    train_l2_loss_df = pd.DataFrame(train_l2_loss, columns=['iteration', 'train_l2_loss'])
    val_cz_loss_df = pd.DataFrame(val_cz_loss, columns=['iteration', 'val_unit_circle_loss'])

    # plot with test
    sns.lineplot(data=train_cz_loss_df, x='iteration', y='train_unit_circle_loss', label='train_unit_circle_loss', ci='sd')
    sns.lineplot(data=train_a_loss_df, x='iteration', y='train_toeplitz_loss', label='train_toeplitz_loss', ci='sd')
    sns.lineplot(data=train_l2_loss_df, x='iteration', y='train_l2_loss', label='train_l2_loss', ci='sd')
    sns.lineplot(data=val_cz_loss_df, x='iteration', y='val_unit_circle_loss', label='val_unit_circle_loss', ci='sd')

    plt.ylim([0, 1])
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(path + ".pdf", dpi=300)
    plt.close()


def heatmap(matrix, highlight_index=None, labels_left=None, labels_right=None, path=None):
    """
    :param matrix: (n x n)
    :param highlight_index: (n x 1)
    :return:
    """
    fig, ax = plt.subplots()
    ax = sns.heatmap(matrix, ax=ax, linewidths=.5, cbar=False,
                     yticklabels=["%.1f" % l for l in labels_left])
    ax.set_xticklabels([])
    if labels_right is not None:
        ax2 = ax.twinx()
        ax2.set_yticks([l for l in labels_right])
        ax2.set_yticklabels(["%.1f" % l for l in labels_right])
    if highlight_index is not None:
        mark_in_heatmap(ax, highlight_index)
    if path is not None:
        plt.savefig(path, dpi=150)
    return ax


def mark_in_heatmap(ax, idxs):
    for idx in idxs:
        idx = idx[::-1]
        ax.add_patch(Rectangle(idx, width=1, height=1, fill=False,
                               edgecolor='green', lw=1, clip_on=False))


def introspection_plot(image: tf.Tensor, beam: tf.Tensor, rot_beam: tf.Tensor, latents: tf.Tensor,
                       angle_energy: tf.Tensor, rnn_encoding: tf.Tensor, lines, angle, gt_angles, name=''):
    fig, ax = plt.subplots(2, 3, figsize=(10, 15))
    fig.suptitle(name, fontsize=16)

    gt_angles = (gt_angles / (2 * math.pi)) * 360.

    beam_eval(image, lines, beam=beam, ax=ax[0, 0], title='Input Beams (0)')
    beam_img = beam_eval(image, lines, beam=rot_beam, ax=ax[0, 1],
                       title='Input Beams (0) Angle %.1f' % gt_angles[0])

    ax[0, 2].imshow(tfa.image.rotate(beam_img, -angle, interpolation='bilinear'))
    angle = (angle / (2 * math.pi)) * 360.
    ax[0, 2].set_title('Rotated by Prediction %.1f' % angle)

    pca_tsne_latent_plot(image, latents, lines, ax=ax[1, 0], width=500, height=500,
                         use_images=False, partial=1, n_components=2, indicator_size=10)

    sns.heatmap(angle_energy, linewidths=.5, ax=ax[1, 1])
    ax[1, 1].set_title('Angle Energy')

    # cut down number of batch elements for visual sake
    if rnn_encoding.shape[0] >= rnn_encoding.shape[1]:
        rnn_encoding = rnn_encoding[:rnn_encoding.shape[1]]
        gt_angles = gt_angles[:rnn_encoding.shape[1]]
    # order the batch elements according to their associated angle
    order = np.argsort(gt_angles.astype(int))
    ax[1, 2].imshow(rnn_encoding[order][..., None])
    # ax[0, 1].set_yticks(list(gt_angles.astype(int)[order]))
    # ax[0, 1].set_yticks([0, 90, 180, 270, 360])
    # ax[0, 1].set_ylabel('GT Angles (Batch)')
    # ax[0, 1].set_xlabel('RNN Encoding')
    ax[1, 2].set_title('RNN Encodings')

    plt.savefig('./data/' + name.replace(' ', '_') + '.pdf', dpi=150)
    plt.close()


def pca(images, latents, lines, partial=100, use_images=True, n_components=16):
    """
    :param images:
    :param latents:
    :param lines:
    :param partial:
    :param use_images: if enabled images are used, otherwise (more cost efficient)
    indicators for the vanilla or rotated versions are used (0,0)
    :param n_components:
    :return:
    """
    pca = PCA(n_components)
    imgs_list, vis_imgs = [], []
    data = zip([images, ], [latents, ]) if partial == 1 else zip(images[:partial], latents[:partial])
    for img, latent in data:
        if use_images:
            img = tf.tile(img, [1, 1, 1, 3]) if img.shape[-1] == 1 else img
            img = img.numpy()
        for i in range(latents.shape[0]):
            for v in range(latents.shape[1]):
                if use_images:
                    _img = copy.deepcopy(img[i])
                    _img[lines[v, 1, :, 0], lines[v, 1, :, 1]] = (1, 0, 0)
                    vis_imgs.append(tf.keras.preprocessing.image.img_to_array(_img))
                else:
                    vis_imgs.append(i)
                imgs_list.append(latent[i, v])
    img_mat = np.array(imgs_list)
    pca_feat = np.array(pca.fit_transform(img_mat))
    return pca_feat, vis_imgs


def pca_tsne_latent_plot(images, latents, lines, ax=None, return_img=False,
                         use_images=True, path='', partial=100, n_components=16, indicator_size=10,
                         indicator_color_a=(0, 192, 192), indicator_color_b=(192, 0, 192),
                         width=1000, height=1000):
    """ This will plot the dimensionally reduced latent manifold.
    Points are replaced by their corresponding images.
    :param images: (batch x 1 x width x height x channels)
    :param latents: (batch x 1 x Beams x hidden)
    :param path: path to save the plot
    """
    pca_feat, vis_imgs = pca(images, latents, lines, partial=partial,
                             use_images=use_images, n_components=n_components)

    tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(pca_feat)
    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    max_dim = 0

    full_image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(full_image)
    for img, x, y in zip(vis_imgs, tx, ty):
        pos = (int((width - max_dim) * x), int((height - max_dim) * y))
        if use_images:
            img = np.array(img * 255, dtype=np.uint8).squeeze()
            tile = Image.fromarray(img)
            full_image.paste(tile, pos, mask=tile.convert('RGBA'))
        else:
            color = indicator_color_a if img == 0 else indicator_color_b
            draw.rectangle((pos[0], pos[1], pos[0]+indicator_size, pos[1]+indicator_size), fill=color)

    if return_img:
        return full_image

    if ax is None:
        plt.figure(figsize=(width // 100, height // 10))
        plt.imshow(full_image)
        plt.axis("off")
        full_image.save(path + '.pdf')
    else:
        ax.imshow(full_image)
        plt.axis("off")
        ax.set_title('PCA t-SNE Latent')


def beam_similarity(sample: dict, lines, distance_matrix: tf.Tensor, normalize=True) -> tuple:
    # extract the gt and opposite gt beam position
    gt_beam_id = tf.cast(tf.where(sample['angle'] == tf.constant(1.))[:, 1], tf.int32)
    ogt_beam_id = tf.math.abs(tf.constant(len(lines) // 2)[None] - gt_beam_id)

    # use the Toeplitz Extractor at that position to extract the corresponding diagonal
    gt_toeplitz = tf.cast([tf.roll(tf.eye(len(lines)), shift=i, axis=0) for i in gt_beam_id], tf.float32)
    ogt_toeplitz = tf.cast([tf.roll(tf.eye(len(lines)), shift=i, axis=0) for i in ogt_beam_id], tf.float32)

    gt_beam_dist = distance_matrix * gt_toeplitz
    ogt_beam_dist = distance_matrix * ogt_toeplitz

    # sum over distances (reminder: all irrelevant ones are zero)
    gt_beam_dist = tf.reduce_sum(gt_beam_dist, axis=(-1, -2))
    ogt_beam_dist = tf.reduce_sum(ogt_beam_dist, axis=(-1, -2))

    # average over batch
    gt_beam_dist = tf.reduce_mean(gt_beam_dist)
    ogt_beam_dist = tf.reduce_mean(ogt_beam_dist)

    # min max normalized similarity
    if normalize:
        toeplitz = tf.cast([tf.roll(tf.eye(len(lines)), shift=i, axis=0) for i in tf.range(len(lines))],  tf.float32)
        energies = tf.reduce_sum(distance_matrix[:, None, ...] * toeplitz[None, ...], axis=(-1, -2))
        # the average similarity respecting the gt aligned ordering
        energies = tf.reduce_mean(tf.matmul(energies, gt_toeplitz), axis=0)
        energy_min, energy_max = tf.reduce_min(energies), tf.reduce_max(energies)
        gt_beam_dist = (gt_beam_dist - energy_min) / (energy_max - energy_min)
        ogt_beam_dist = (ogt_beam_dist - energy_min) / (energy_max - energy_min)

    return gt_beam_dist, ogt_beam_dist
