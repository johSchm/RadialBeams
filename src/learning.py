import math
import tensorflow as tf
from tqdm import tqdm



# def get_gt_beam_energy(epoch: int):
#     # reason: the dot product dampens all signals (zero) apart from the target dimension
#     sigma = 1/(epoch+1) * 100 #
#     x = tf.range(-(n_beams // 2), n_beams // 2 + 1, dtype=tf.float32)
#     kernel = 1 / (sigma * tf.math.sqrt(2*math.pi)) * tf.math.exp(-x**2/(2*sigma**2))
#     kernel = kernel / tf.reduce_sum(kernel)
#     return kernel

# def dkl_gaussian(pred, stdd=1.):
#     x = tf.range(-(n_beams // 2), n_beams // 2 + 1, dtype=tf.float32)
#     # 1/((x \sigma \sqrt{2 \pi})
#     kernel = 1 / (x * stdd * tf.math.sqrt(2 * math.pi))
#     # \exp{-(\ln x-\mu)^2)/(2 \sigma^2}}
#     kernel = kernel * tf.math.exp(-(tf.math.log(x)) / (2 * stdd ** 2))
#     kernel = kernel / tf.reduce_sum(kernel)
#     # \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right)
#     return tf.reduce_sum(tf.math.log(pred / kernel), axis=-1)

def cyclic_angle_loss(complex_angle_pred):
    complex_angle_true = tf.fill((complex_angle_pred.shape[0],), math.pi/2)
    return ((tf.math.sin(complex_angle_true) - complex_angle_pred[:, 1]) ** 2
            + (tf.math.cos(complex_angle_true) - complex_angle_pred[:, 0]) ** 2)

def get_laplace(n_beams, scale=10):
    x = tf.range(-(n_beams // 2), n_beams // 2 + 1, dtype=tf.float32)
    return tf.math.exp(-tf.math.abs(x) / scale) / (2 * scale)

def training(model, train_dataset, val_dataset, test_dataset, optimizer, lines, angles,
             epochs=128, name='', continuous=False, prior='off'):

    pbar = tqdm(range(epochs), desc='---')
    for e in pbar:

        # validation, before training to log also the init behaviour of the model
        val_circle_loss = []
        for i, sample in enumerate(val_dataset):
            x = tf.concat([sample['beam'][:, None], sample['beam_rot'][:, None]], axis=1)
            pred_facts, pred_angle, conv_latents, gnn_latents, distance_matrix, \
            x1_emb, x2_emb, angle_energy, rnn_encoding = model(x)
            unit_circle_loss, _ = loss_func(pred_angle, pred_facts, angles, sample['angle'], continuous=continuous)
            val_circle_loss.append(unit_circle_loss)

        if test_dataset is not None:
            # testing / deployment
            test_circle_loss = []
            for i, sample in enumerate(test_dataset):
                # duplicate the second (rotated) augmentation image and ignore the angle branch output
                # (batch x augmentation x size_vector_field x proximity x pixel_count_per_vector x channels)
                pred_facts, pred_angle, conv_latents, gnn_latents, distance_matrix, \
                x1_emb, x2_emb, angle_energy, rnn_encoding = model(
                    tf.tile(sample['beam_rot'][:, None, ...], [1, 2, 1, 1, 1, 1]))
                unit_circle_loss, _ = loss_func(pred_angle, pred_facts, angles, sample['angle'], continuous=continuous)
                test_circle_loss.append(unit_circle_loss)
        else:
            test_circle_loss = 0.

        # training
        train_circle_loss = []
        for i, sample in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                x = tf.concat([sample['beam'][:, None], sample['beam_rot'][:, None]], axis=1)
                pred_facts, pred_angle, conv_latents, gnn_latents, distance_matrix, \
                x1_emb, x2_emb, angle_energy, rnn_encoding = model(x)
                unit_circle_loss, toeplitz_loss = loss_func(pred_angle, pred_facts, angles,
                                                            sample['angle'], continuous=continuous)

                if prior == 'off':
                    loss = unit_circle_loss
                elif prior == 'only':
                    loss = toeplitz_loss
                elif prior == 'linear':
                    k = e / (epochs - 1)
                    loss = k * unit_circle_loss + (1 - k) * toeplitz_loss
                else:
                    loss = unit_circle_loss + toeplitz_loss

            grads_model = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads_model, model.trainable_variables))
            train_circle_loss.append(unit_circle_loss)

        pbar.set_description("Training loss {0:.3f} | Validation loss {1:.3f} | Testing loss {2:.3f}".format(
            tf.reduce_mean(train_circle_loss), tf.reduce_mean(val_circle_loss), tf.reduce_mean(test_circle_loss)))

    # tf.keras.models.save_model(model, 'model/' + name + '.h5py', include_optimizer=False)
    model.save_weights('model/' + name + '.h5', overwrite=True)


def loss_func(pred_angle: tf.Tensor, pred_facts, angles, gt_angles, continuous=False) -> tuple:
    """ Regression since this respects the spatial angle information.
    That is, if the prediction is 30 degree but the true is 35 degree,
    a regression will respect the closeness which is ignored by a categorical loss.

    DO NOT USE tf.reduce_sum(angle_factor_distr * angles[None, :], axis=-0)
    since this introduces symmetries, like 0.5 * 10 + 0.5 * 20 = 0.0 * 50 + 0.5 * 20
    """
    d = tf.constant(0.)
    if not continuous:
        # invert the angle since we would like to rotate back
        # from an optimization perspective that shouldn't make a difference
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0, axis=-1)
        # log to encourage exploration
        d = cce(gt_angles, pred_facts)
        gt_angles = (tf.reduce_sum(angles[None, :] * gt_angles, axis=-1) * math.pi) / 180.
    a = (tf.math.sin(gt_angles) - pred_angle[:, 1]) ** 2 + (tf.math.cos(gt_angles) - pred_angle[:, 0]) ** 2
    return tf.reduce_mean(a), tf.reduce_mean(d)
