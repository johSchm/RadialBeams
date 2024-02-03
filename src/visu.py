import copy

import matplotlib.pyplot as plt
import numpy as np
import imutils
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw
from sklearn.manifold import TSNE
import tensorflow as tf
import math
import tensorflow_addons as tfa
import wandb
from matplotlib.patches import Rectangle


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
