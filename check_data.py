"""
Check TianChi Data Test2
"""

import os
import numpy as np
import torch
import viewer
import argparse

parser = argparse.ArgumentParser(description='PyTorch TianChi Nodule Detector')
parser.add_argument('-i', '--index', default=31, type=int, metavar='N')
args = parser.parse_args()


viewer.FIG_SIZE = (16, 16)  # figure size

# view scan to check for interpolated data
data_dir = '/ssd_1t/TianChi/npy_data_zoomed/train'
label_dir = '/ssd_1t/TianChi/npy_nodule_mask_zoomed/train'
result_file = '/home/wangd/program/3Drcnn/results/res18_20170706-2317/test_result.ckpt'
results = torch.load(result_file)['result_dict']
seriesuids = sorted(results.keys())

uid = seriesuids[args.index]
print(uid)

label = dict(np.load(os.path.join(label_dir, uid + '.npz')).items())
bbox_label = np.concatenate([label['nodule_voxel_coord'] - label['nodule_voxel_diameter'].reshape((-1, 1)) / 2. - 5,
                             label['nodule_voxel_coord'] + label['nodule_voxel_diameter'].reshape((-1, 1)) / 2. + 5], 1)
attrs_label = ['+'] * len(bbox_label)
colors_label = ['red'] * len(bbox_label)

vol = np.load(os.path.join(data_dir, uid + '.npy'))
vol[vol < -1024] = -1024
vol[vol > 300] = 300

coord = results[uid]
coord = coord[coord[:, 0] > 0.4]
bbox_pred = np.concatenate([coord[:, 1:4] - coord[:, 4:5] / 2. - 5, coord[:, 1:4] + coord[:, 4:5] / 2. + 5], 1)
bbox_pred = np.rint(bbox_pred)
conf = coord[:, 0]
attrs_pred = ['{:.3f},{:.3f}'.format(x[0], x[4] * label['spacing'][0] * (label['vol_shape'][0] - 1) / (label['vol_zoomed_shape'][0] - 1 )) for x in coord]
colors_pred = ['yellow'] * len(coord)


viewer.view_scan(vol,
                 bbox_list=np.concatenate([bbox_label, bbox_pred], 0),
                 attr_list=attrs_label+attrs_pred,
                 color_list=colors_label+colors_pred)


# # print scan to check for interpolated data
# data_dir = '/data-174/TianChi/npy_data_zoomed/train'
# label_dir = '/data-174/TianChi/npy_nodule_mask_zoomed/train'
# output_dir = '/data-174/TianChi/check_data/interpolated_version'

# for data_file in os.listdir(data_dir):
# 	scan_name = os.path.splitext(data_file)[0]
# 	print scan_name
# 	vol = np.load(os.path.join(data_dir, data_file), mmap_mode='r')
# 	with np.load(os.path.join(label_dir, '%s.npz' %scan_name)) as labels:
# 		coords = labels['nodule_voxel_coord']
# 		diameters = labels['nodule_voxel_diameter']
# 		bbox_list = [B.center2bbox(coord, 40) for coord,diameter in zip(coords, diameters)]
# 		for bbox_id, bbox in enumerate(bbox_list):
# 			subvol = B.create_subvol(vol, bbox, padding_value=-1024).copy()
# 			subvol[subvol<-1024] = -1024
# 			subvol[subvol > 300] = 300
# 			title = '%s_%d' % (scan_name, bbox_id)
# 			viewer.vis_slices(subvol, ncols=8, title=str(diameters[bbox_id]), path=os.path.join(output_dir, title), color=False)
