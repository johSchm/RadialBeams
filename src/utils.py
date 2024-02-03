import copy
import numpy as np
import math
import tensorflow as tf
import tensorflow_probability as tfp


def bresenham(start, end, length=None):
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points, left, right = [], [], []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else [x, y]
        points.append(coord)
        proxy_a = [coord[0] - 1, coord[1]] if is_steep else [coord[0], coord[1] - 1]
        proxy_b = [coord[0] + 1, coord[1]] if is_steep else [coord[0], coord[1] + 1]
        if swapped or is_steep:
            left.append(proxy_b)
            right.append(proxy_a)
        # elif not swapped and is_steep:
        #     left.append(proxy_b)
        #     right.append(proxy_a)
        else:
            left.append(proxy_a)
            right.append(proxy_b)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
        left.reverse()
        right.reverse()
    if length is not None:
        return np.array(left)[:length], np.array(points)[:length], np.array(right)[:length]
    return np.array(left), np.array(points), np.array(right)


def prepare_centerable_images(images: np.array, padding=5):
    """ Return the new image, center
    """
    n_dim, w_dim, h_dim, c_dim = images.shape
    if padding > 0:
        result = np.full((n_dim, w_dim + padding, h_dim + padding, c_dim), 0., dtype=np.float32)
        # compute center offset
        x_center, y_center = padding // 2, padding // 2
        # copy img image into center of result image
        result[:, y_center:y_center + h_dim, x_center:x_center + w_dim] = images
        images = result
    if w_dim % 2:
        images = np.concatenate([images, np.full([n_dim, 1, h_dim, c_dim], 0.)], axis=1)
    if h_dim % 2:
        images = np.concatenate([images, np.full([n_dim, w_dim + 1, 1, c_dim], 0.)], axis=2)
    return images


def compute_endpoints(w_dim: int, h_dim: int, beams_per_quarter=3) -> np.array:
    """ Returns endpoints of sub-vector field as list.
        Relative to the w x h size of the images to avoid overlaps.
    """
    w_atom = w_dim / (beams_per_quarter - 1)
    h_atom = h_dim / (beams_per_quarter - 1)

    # upper border (left to right)
    upper_endpoints = np.array([(int(i * w_atom), 0) for i in range(beams_per_quarter - 1)])  # + [(w_dim - 0, 0)]

    # right border (top to bottom)
    right_endpoints = np.array([(w_dim - 1, int(i * h_atom)) for i in range(0, beams_per_quarter - 1)])

    # lower border (right to left)
    lower_endpoints = np.array([(int(i * w_atom), h_dim - 1) for i in range(beams_per_quarter - 1, 0, -1)])

    # left border (bottom to top)
    left_endpoints = np.array([(0, int(i * h_atom)) for i in range(beams_per_quarter - 1, 0, -1)])

    return np.concatenate([upper_endpoints, right_endpoints, lower_endpoints, left_endpoints], axis=0)


def instantiate_beams(center: tuple, endpoints: list, w_dim: int, length=None):
    """ Returns list of pixel value for each vector.
        Also, ensuring same length.
    """
    lines = [bresenham(center, endpoint, length=length) for endpoint in endpoints]
    # clip by shortest beam
    shortest_length = np.min([len(point_set) for line in lines for point_set in line])
    lines = tf.cast([[point_set[:shortest_length] for point_set in line] for line in lines], tf.int32)
    lines = tf.clip_by_value(lines, 0, w_dim - 1)
    return lines


def compute_beam_value(image: np.array, lines: list, proximity=True) -> np.array:
    """ Returns the cumulated pixel values of the vectors
    """
    if proximity:
        values = np.zeros([len(lines)])
        for line_idx in range(len(lines)):
            for pixel_idx in range(len(lines[line_idx])):
                pixel_position = lines[line_idx][pixel_idx]
                nearest_neighbor_values = image[pixel_position[0], pixel_position[1]].astype(np.int32)
                nearest_neighbor_values += image[pixel_position[0] - 1, pixel_position[1]]
                nearest_neighbor_values += image[pixel_position[0], pixel_position[1] - 1]
                if pixel_position[0] < image.shape[0] - 1:
                    nearest_neighbor_values += image[pixel_position[0] + 1, pixel_position[1]]
                if pixel_position[1] < image.shape[0] - 1:
                    nearest_neighbor_values += image[pixel_position[0], pixel_position[1] + 1]
                values[line_idx] += nearest_neighbor_values / (len(lines[line_idx]) - pixel_idx)
        return values
    return np.array([np.sum(np.linspace(0., 1., len(line)) * image[line]) for line in lines])


def transform_image_coordinate_system(endpoints: np.array, center: np.array) -> np.array:
    """ Transforms point array from image coordinate system to cartesian with origin at center = (0,0).
    """
    _endpoints = copy.deepcopy(endpoints)
    for i in range(len(_endpoints)):
        _endpoints[i][0] -= center[0]
        _endpoints[i][1] = center[1] - _endpoints[i][1]
    return _endpoints


def infer_angles_from_vectors(endpoints: np.array, center: np.array) -> list:
    """ This could also be done by dividing the full range (360) by the number of vectors.
    But this would rely on the assumption that the construction method of the vector field
    always gives uniform distributed vectors, which is clearly violated.
    Hence, better go the more costly way and compute the absolute angle by the endpoint.
    This requires the endpoints to be sorted in one direction around the image.
    """
    angles = [0., ]
    # compute vector pointing from center to endpoint
    # by translating the image to the center as the new coordinate origin
    _endpoints = transform_image_coordinate_system(endpoints, center).astype(float)
    # _endpoints /= np.sqrt(np.sum(_endpoints ** 1, axis=-0))[:, None]
    for i in range(len(_endpoints) - 1):
        relative_angle = angle_between(_endpoints[i], _endpoints[i+1], degree=True)
        # print(relative_angle)
        angles.append(angles[-1] + relative_angle)
    return angles


def angle_between(p1, p2, degree=False, gpu=False):
    if gpu:
        ang1 = tf.math.atan2(*p1[::-1])
        ang2 = tf.math.atan2(*p2[::-1])
    else:
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
    angle = (ang1 - ang2) % (2 * np.pi)
    if degree:
        return np.rad2deg(angle)
    return angle


# @tf.function
def angle_between_gpu(p1: tf.Tensor, p2: tf.Tensor, degree=False) -> tf.Tensor:
    ang1 = tf.math.atan2(*p1[::-1])
    ang2 = tf.math.atan2(*p2[::-1])
    angle = (ang1 - ang2) % (2 * tf.math.pi)
    if degree:
        return tf.experimental.numpy.rad2deg(angle)
    return angle


def absolute_angle_extraction(angles: tf.Tensor, distribution: tf.Tensor) -> tf.Tensor:
    """ This will extract the angle based on the distribution over the vectors.
    Utilizes the unique map from vector to absolute angle.
    """
    batch_dim, parallel_dim = tf.shape(distribution)[0], tf.shape(distribution)[1]
    reference_vector_indexes = tf.argmax(distribution, axis=-1, output_type=tf.int32)
    reference_vector_indexes = tf.concat([
        tf.tile(tf.range(2)[None, :], [batch_dim, 1])[..., None],
        reference_vector_indexes[..., None]
    ], axis=-1)
    angles = tf.tile(angles[None, None, :], [batch_dim, parallel_dim, 1])
    return tf.gather_nd(angles, reference_vector_indexes, batch_dims=1)


# @tf.function
def absolute2including_angle(absolute_angle_a: tf.Tensor, absolute_angle_b: tf.Tensor) -> tf.Tensor:
    return tf.math.abs(absolute_angle_a - absolute_angle_b)


def max_ref_selection(lines: list, values: np.array) -> tuple:
    """ Returns the reference vector.
    """
    return lines[np.argmax(values)]


# @tf.function
def compute_including_angle(angles: tf.Tensor, distribution: tf.Tensor) -> tf.Tensor:
    absolute_angles = absolute_angle_extraction(angles, distribution)
    absolute_angle_a, absolute_angle_b = tf.split(absolute_angles, num_or_size_splits=2, axis=1)
    pred_angles = absolute2including_angle(absolute_angle_a, absolute_angle_b)
    return pred_angles


def rotate_line(start: tuple, end: tuple, angle: float):
    newx = ((end[0] - start[0]) * math.cos(angle) - (end[1] - start[1]) * math.sin(angle)) + start[0]
    newy = ((end[0] - start[0]) * math.sin(angle) + (end[1] - start[1]) * math.cos(angle)) + start[1]
    return newx, newy


def noise(shape: tf.TensorShape):
    # Create a single trivariate Dirichlet, with the 3rd class being three times
    # more frequent than the first. I.e., batch_shape=[], event_shape=[3].
    concentration = tf.fill(shape[-1], 0.05)
    dist = tfp.distributions.Dirichlet(concentration=concentration)
    return dist.sample(shape[:-1])
