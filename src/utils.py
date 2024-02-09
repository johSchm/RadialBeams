import copy
import numpy as np
import math
import tensorflow as tf
# import tensorflow_probability as tfp
from scipy.ndimage import map_coordinates
import numpy as np
import tensorflow_addons as tfa


def create_circular_mask(height, width, center=None, radius=None):
    if center is None:
        center = [height // 2, width // 2]
    if radius is None:
        radius = min(center[0], center[1], height - center[0], width - center[1])

    y, x = np.ogrid[:height, :width]
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    return mask.astype(np.float32)


def apply_circular_mask(image):

    center = [image.shape[0] // 2, image.shape[1] // 2]
    radius = min(center[0], center[1])

    height, width, _ = image.shape

    mask = create_circular_mask(height, width, center, radius)
    mask = np.expand_dims(mask, axis=-1)

    masked_image = image * mask

    return tf.convert_to_tensor(masked_image, dtype=tf.float32)


def get_endpoints(radius: int, num_points: int) -> np.ndarray:
    k = np.arange(num_points)
    x = radius * np.cos(2 * k * np.pi / num_points)
    y = radius * np.sin(2 * k * np.pi / num_points)
    return np.stack([x, y], axis=-1) + radius


def cartesian_to_polar_grid(image_grid, image_size):
    """ Computes the angle of each pixel vector w.r.t. the center of the image.
    Therefore, the image_grid is assumed to be [-image_size/2, image_size/2].
    """
    x = image_grid[..., 0]
    y = image_grid[..., 1]

    theta = tf.atan2(y, x)
    # theta = (theta - tf.reduce_min(theta)) / (tf.reduce_max(theta) - tf.reduce_min(theta))
    # theta *= image_size-1
    # print(tf.reduce_min(theta), tf.reduce_max(theta))

    r = tf.sqrt(x ** 2 + y ** 2)
    # r = (r - tf.reduce_min(r)) / (tf.reduce_max(r) - tf.reduce_min(r))
    # r *= image_size-1
    # print(tf.reduce_min(r), tf.reduce_max(r))
    x = r * tf.cos(theta) + image_size / 2
    y = r * tf.sin(theta) + image_size / 2

    x = tf.clip_by_value(x, 0, image_size - 1)
    y = tf.clip_by_value(y, 0, image_size - 1)

    polar_coordinates = tf.stack([x, y], axis=-1)
    return tf.cast(polar_coordinates, tf.int32)


def make_grid(h, w):
    X, Y = tf.meshgrid(tf.linspace(-1., 1., h), tf.linspace(-1., 1., w))
    X = tf.reshape(X, (1, -1))
    Y = tf.reshape(Y, (1, -1))
    grid = tf.concat([X, Y], axis=0)
    return grid


def get_pixel_value(im, x, y):
    b, h, w, c = im.shape
    x, y = tf.cast(x, 'int32'), tf.cast(y, 'int32')
    batch_idx = tf.range(0, b)[:, None, None]
    b = tf.tile(batch_idx, (1, h, w))
    indices = tf.stack([b, y, x], -1)
    return tf.gather_nd(im, indices)


def interpolate(im, x, y):
    b, h, w, c = im.shape
    max_y, max_x = h - 1., w - 1.

    # finds neighbors
    x = 0.5 * ((x + 1.) * max_x)
    y = 0.5 * ((y + 1.) * max_y)
    x0, y0 = tf.floor(x), tf.floor(y)
    x1, y1 = x0 + 1., y0 + 1.
    x0, y0 = tf.clip_by_value(x0, 0., max_x), tf.clip_by_value(y0, 0., max_y)
    x1, y1 = tf.clip_by_value(x1, 0., max_x), tf.clip_by_value(y1, 0., max_y)

    # get neighbor pixels
    Ia = get_pixel_value(im, x0, y0)
    Ib = get_pixel_value(im, x0, y1)
    Ic = get_pixel_value(im, x1, y0)
    Id = get_pixel_value(im, x1, y1)

    # calculate distance to neighbors
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # weighted sum over neighbors
    I = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id
    return I


def log_polar_transform(x, radius_factor=tf.sqrt(2.)):
    '''rho(x) ,theta(y): [-1, 1]
    theta : [-1 * pi + pi, 1 * pi + pi] = [0, 2pi]
    rho: [log(-1 * h/2 + h/2), log(1 * h/2 + h/2)] --> [1, logh]
    r: exp(rho)-1 -->[0, h-1]
    r: normalize --> r/(h-1) -->[0, 1]
    x = e^rho * cos(theta)
    y = e^rho & sin(theta)
    '''
    b, h, w, c = x.shape
    grid = make_grid(h, w)  # (2, hw), represent log-polar coordinate system
    grid = tf.repeat(grid[None, ...], b, axis=0)  # (b, 2, hw)
    X, Y = grid[:, 0], grid[:, 1]

    # theta
    theta = (Y + 1) * math.pi  # [0, 2pi]

    # radius
    maxR = max(h, w) * radius_factor
    r = tf.exp((X + 1) / 2 * tf.math.log(maxR))  # [1, h]
    r = (r - 1) / (maxR - 1)  # [0, h]-->[0, 1]
    r = r * (maxR / h)  # scale factorize

    # map to cartesian coordinate system
    xs = tf.reshape(r * tf.math.cos(theta), [b, h, w])
    ys = tf.reshape(r * tf.math.sin(theta), [b, h, w])
    output = interpolate(x, xs, ys)
    output = tf.reshape(output, [b, h, w, c])
    return output


def grid_sample(image, warp):
    warp = warp + 0.5
    image = tf.pad(image, ((0, 0), (1, 1), (1, 1), (0, 0)))
    warp_shape = tf.shape(warp)
    flat_warp = tf.reshape(warp, (warp_shape[0], -1, 2))
    flat_sampled = tfa.image.interpolate_bilinear(image, flat_warp, indexing="xy")
    output_shape = tf.concat((warp_shape[:-1], tf.shape(flat_sampled)[-1:]), 0)
    return tf.reshape(flat_sampled, output_shape)


def polar_transform(img, origin=None, radius=None, output=None):
    """ Transformation to polar coordinates.
    :param img: (batch x height x width x channel)
    """
    img_shape_tensor = tf.convert_to_tensor([img.shape[1], img.shape[2]], dtype=tf.float32)
    if origin is None:
        origin = img_shape_tensor / 2 - 0.5
    if radius is None:
        radius = tf.reduce_sum(img_shape_tensor ** 2) ** 0.5 / 2
    if output is None:
        output = tf.zeros([tf.cast(tf.math.round(radius), dtype=tf.int32),
                           tf.cast(tf.math.round(radius * 2 * math.pi), dtype=tf.int32)], dtype=img.dtype)
    elif isinstance(output, tuple):
        output = tf.zeros(output, dtype=img.dtype)
    out_h, out_w = output.shape
    rs = tf.linspace(0., radius, out_h)
    ts = tf.linspace(0., math.pi * 2, out_w)
    grid = tf.stack([  # 2 x beam_len x n_beams
        rs[:, None] * tf.cos(ts) + origin[1],  # x
        rs[:, None] * tf.sin(ts) + origin[0]  # y
    ])
    # batch x beam_len x n_beams x 2
    grid = tf.transpose(grid, (1, 2, 0))[None]
    return grid_sample(img, grid)

def polar_transform_batch(images, o=None, r=None, output=None, order=1, cont=0):
    return tf.map_fn(
        lambda image: polar_transform(image, o=o, r=r, output=output, order=order),
        images, dtype=images.dtype)

def polar_transform_inv(image, o=None, r=None, output=None, order=1, cont=0):
    # https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/3
    output_image = []
    image = image.numpy()
    for c in range(image.shape[-1]):
        img = image[..., c]
        if r is None: r = img.shape[0]
        if output is None:
            output = np.zeros((r*2, r*2), dtype=img.dtype)
        elif isinstance(output, tuple):
            output = np.zeros(output, dtype=img.dtype)
        if o is None: o = np.array(output.shape)/2 - 0.5
        out_h, out_w = output.shape
        ys, xs = np.mgrid[:out_h, :out_w] - o[:,None,None]
        rs = (ys**2+xs**2)**0.5
        ts = np.arccos(xs/rs)
        ts[ys<0] = np.pi*2 - ts[ys<0]
        ts *= (img.shape[1]-1)/(np.pi*2)
        map_coordinates(img, (rs, ts), order=order, output=output)
        output_image.append(output)
    return tf.stack(output_image, axis=-1)

def inverse_log_polar_transform(x):
    b, h, w, c = x.shape
    grid = make_grid(h, w)  # (2, hw), represent log-polar coordinate system
    grid = tf.repeat(grid[None, ...], b, axis=0)  # (b, 2, hw)
    X, Y = grid[:, 0], grid[:, 1]

    rs = tf.sqrt(X ** 2 + Y ** 2) / tf.sqrt(2.)
    ts = (tf.atan2(-Y, -X)) / math.pi  # [-1., 1.]

    rs = tf.reshape(rs, [b, h, w])
    ts = tf.reshape(ts, [b, h, w])
    output = interpolate(x, rs, ts)
    output = tf.reshape(output, [b, h, w, c])
    return output


def create_circle_matrix(height, width, radius, thickness):
    matrix = np.ones((height, width, 1))
    y, x = np.ogrid[:height, :width]

    # Equation of a circle centered at (h/2, w/2) with specified radius and thickness
    circle_mask = ((x - width / 2) ** 2 + (y - height / 2) ** 2 <= (radius) ** 2) & \
                  ((x - width / 2) ** 2 + (y - height / 2) ** 2 >= (radius - thickness) ** 2)

    # Set the values inside the circle to 0 (black)
    matrix[circle_mask] = 0

    # Equation of a circle centered at (h/2, w/2) with specified radius and thickness
    circle_mask = ((x - width / 2) ** 2 + (y - height / 2) ** 2 <= (radius / 2) ** 2) & \
                  ((x - width / 2) ** 2 + (y - height / 2) ** 2 >= (radius / 2 - thickness) ** 2)

    # Set the values inside the circle to 0 (black)
    matrix[circle_mask] = 0.5
    return matrix


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
    """ TODO deprecated -> remove
        Returns endpoints of sub-vector field as list.
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


# def noise(shape: tf.TensorShape):
#     # Create a single trivariate Dirichlet, with the 3rd class being three times
#     # more frequent than the first. I.e., batch_shape=[], event_shape=[3].
#     concentration = tf.fill(shape[-1], 0.05)
#     dist = tfp.distributions.Dirichlet(concentration=concentration)
#     return dist.sample(shape[:-1])
