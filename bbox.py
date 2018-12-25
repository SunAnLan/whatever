""" Implementation of helpers to deal with bounding boxes
bbox: [min_dim0, min_dim1, min_dim2, max_dim0, max_dim1, max_dim2]
center: [dim0, dim1, dim2]
"""

import numpy as np


def regularize(x):
    """
    Regularize the array by rounding and casting into integer
    """
    return np.round(x).astype(np.int16)


def bbox2center(bbox):
    bbox = np.array(bbox, dtype=np.float32)
    center = 0.5 * np.array([(bbox[0] + bbox[3]), (bbox[1] + bbox[4]), (bbox[2] + bbox[5])])
    return regularize(center)


def center2bbox(center, diameter):
    """
    Converting given center into bounding box
    :param center: (3,)
    :param diameter: () or (3,)
    :return: (6,) bounding box
    """
    diameter = np.array(diameter)
    if len(diameter.shape) == 0:
        diameter = np.repeat(diameter, 3)

    center = regularize(center)
    left = np.floor(diameter / 2).astype(np.int16)
    right = np.ceil(diameter / 2).astype(np.int16)

    bbox = np.array([center[0] - left[0], center[1] - left[1], center[2] - left[2],
                     center[0] + right[0], center[1] + right[1], center[2] + right[2]])

    return bbox


def in_bbox(center, bbox):
    return bbox[0] <= center[0] <= bbox[3] \
           and bbox[1] <= center[1] <= bbox[4] \
           and bbox[2] <= center[2] <= bbox[5]


def create_subvol(vol, bbox, padding_value=-1024):
    """
    Create a sub-volume of a given volume according to bbox
    Similar to np.pad, but faster
    :param vol: given volume, only required to support [] operator. Thus, available for np.load or h5py.dataset
    :param bbox: bounding box to crop
    :param padding_value: value to pad
    :return: a sub-volume
    """
    bbox = regularize(bbox)
    valid_min = np.max([[0, 0, 0], bbox[0:3]], axis=0)
    valid_max = np.min([vol.shape, bbox[3:6]], axis=0)
    # print(valid_min, valid_max)

    padding_min = np.max([[0, 0, 0], -bbox[0:3]], axis=0)
    padding_max = np.max([[0, 0, 0], bbox[3:6] - vol.shape], axis=0)
    # print(padding_min, padding_max)

    # t = time.time()
    subvol = vol[valid_min[0]:valid_max[0], valid_min[1]:valid_max[1], valid_min[2]:valid_max[2]]
    # print(time.time()-t)

    for dim in range(subvol.ndim):
        if padding_min[dim] == 0 and padding_max[dim] == 0:
            continue

        padding_shape = list(subvol.shape)
        padding_shape[dim] = padding_min[dim]
        padding_left = np.zeros(shape=padding_shape,
                                dtype=subvol.dtype) + padding_value
        padding_shape[dim] = padding_max[dim]
        padding_right = np.zeros(shape=padding_shape,
                                 dtype=subvol.dtype) + padding_value

        subvol = np.concatenate((padding_left, subvol, padding_right), axis=dim)

    # pad_width = list()
    # for dim in range(3):
    #     pad_width.append((padding_min[dim], padding_max[dim]))
    #
    # # print pad_width
    #
    # subvol = np.pad(subvol, pad_width, 'constant', constant_values=padding_value)

    return subvol
