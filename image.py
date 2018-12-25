""" Implementation of helpers to augment data
Leverage the interfaces provided by keras, which is based on scipy.ndimage
Notice: channel axis is set to 0 as default, and padding value is 0

Additional libraries:
    numpy
    scipy.ndimage
"""

import numpy as np
import scipy.ndimage as ndi

constant_value = 0.


# image pre-processing
def interpolation(vol, ori_spacing, new_spacing=None):
    if new_spacing is None:
        new_spacing = np.array([ori_spacing[0], ori_spacing[1], np.max(ori_spacing[0:2])])

    ori_shape = vol.shape
    vol = ndi.map_coordinates(
        vol,
        np.mgrid[0:ori_shape[0]:(new_spacing[0]/ori_spacing[0]),
            0:ori_shape[1]:(new_spacing[1]/ori_spacing[1]),
            0:ori_shape[2]:(new_spacing[2]/ori_spacing[2])],
        order=1,
        mode='nearest'
    )

    assert ori_shape[0:2] == vol.shape[0:2]

    return vol, new_spacing


# source code from keras.preprocessing.image
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


# implemented helpers
def normalize(x, v_min=-1024., v_max=800.):
    x[x < v_min] = v_min
    x[x > v_max] = v_max
    return (x - v_min) / (v_max - v_min)


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def crop(x, crop_bbox):
    return x[crop_bbox[0]:crop_bbox[3], crop_bbox[1]:crop_bbox[4],crop_bbox[2]:crop_bbox[5]]


def rotate(x, theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[0], x.shape[1]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, 0, 'constant', constant_value)
    return x


def shift(x, dh, dw):
    translation_matrix = np.array([[1, 0, dh],
                                   [0, 1, dw],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, 0, 'constant', constant_value)
    return x


def zoom(x, zx, zy):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[0], x.shape[1]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, 0, 'constant', constant_value)
    return x

# please implement the random_augmentation according to your problem
# example:
# def random_augmentation(x, choice):
#     if x.ndim == 2:
#         x = np.expand_dims(x,-1)
#     else:
#         assert x.ndim == 3, 'ndim of input is expected to be 2 or 3.'
#
#     assert choice in range(SUM_AUG_TYPE)
#
#     if choice == 0:
#         return flip_axis(x, axis=0)
#     elif choice == 1:
#         return flip_axis(x, axis=1)
#     elif choice == 2:
#         return flip_axis(x, axis=2)
#     elif choice == 3:
#         dh, dw = np.random.uniform(-8, 8, 2)
#         return random_shift(x, dh, dw)
#     elif choice == 4:
#         theta = np.pi / 180 * np.random.uniform(-5, 5)
#         return random_rotation(x, theta)
#     elif choice == 5:
#         zx, zy = np.random.uniform(0.875, 1.125, 2)
#         return random_shift(x, zx, zy)