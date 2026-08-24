import argparse
import numpy as np
import os
from npy_append_array import NpyAppendArray
import math
import torch.nn.functional as F
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--split')
args = parser.parse_args()
    
# point to the directory structure in step 8
data_path = args.data_path
which_set = args.split

manifest_path = os.path.join(data_path, f'discrete_speech_{which_set}_clsattnbndkmeans')
data = np.load(os.path.join(manifest_path, 'data.npy'), mmap_mode='r')

with open(os.path.join(manifest_path, 'data.lengths'), 'r') as f:
    data_lens_f = f.readlines()
    
data_lens = []

for line in data_lens_f:
    data_lens.append(int(line.strip()))
    
offsets = [0] + list(np.cumsum(data_lens)[:-1])

filenames = []
with open(os.path.join(manifest_path, 'file_list.txt'), 'r') as f:
    file_lines = f.readlines()
for file_line in file_lines:
    filenames.append(file_line.strip())

with open(os.path.join(manifest_path, 'data_dict_bnd.pkl'), 'rb') as fb:
    boundary_dict = pickle.load(fb)
    

seg_feats_path = os.path.join(data_path, 'seg_feats')

os.makedirs(seg_feats_path, exist_ok=True)
if os.path.exists(os.path.join(seg_feats_path, f'{which_set}_0_1.npy')):
    os.remove(os.path.join(seg_feats_path, f'{which_set}_0_1.npy'))
npaa = NpyAppendArray(os.path.join(seg_feats_path, f'{which_set}_0_1.npy'))


with open(os.path.join(seg_feats_path, f'{which_set}_0_1.len'), 'w') as fw, open(os.path.join(seg_feats_path, f'{which_set}_bnd_errors.txt'), 'w') as f_err:
    for idx, filename in enumerate(filenames):
        # print(filename, data_lens[idx])
        if data_lens[idx] == 0:
            print('bad file encountered')
            
        curr_data = data[offsets[idx]: offsets[idx] + data_lens[idx]]
        curr_seg_feats = []
        boundaries = boundary_dict[filename]
        
        boundary_locs = np.where(boundaries['seg_bound'] == 1)[0]
        # print(boundary_locs, len(curr_data))
        if len(boundary_locs) == 0:
            boundary_locs = np.arange(0, 2 * (len(curr_data)), 26).astype('int32')
            f_err.write(filename + ' empty boundaries found!' + '\n')
        if len(boundary_locs) == 1:
            boundary_locs = np.arange(0, 2 * (len(curr_data)), 26).astype('int32')
            f_err.write(filename + ' length-one boundaries found!' + '\n')
        if len(boundary_locs) == 1:
            boundary_locs = np.array([0, 2 * (len(curr_data))]).astype('int32')
            f_err.write(filename + ' length-one boundaries found but due to short utt!' + '\n')
        
        boundary_locs = boundary_locs // 2
        boundary_locs = sorted(list(boundary_locs))
        # print(boundary_locs)
        if boundary_locs[0] != 0:
            boundary_locs = [0] + boundary_locs
        while len(boundary_locs) > 0 and boundary_locs[-1] >= len(curr_data) - 1:
            # print(boundary_locs)
            boundary_locs = boundary_locs[:-1]
            
        if len(boundary_locs) == 0 or boundary_locs[0] != 0:
            boundary_locs = [0] + boundary_locs
            
        boundary_locs = boundary_locs + [len(curr_data)]
        # print(boundary_locs)
            
        for start_frame, end_frame in zip(boundary_locs[:-1], boundary_locs[1:]):
            # print(start_frame, end_frame, len(curr_data))
            if start_frame != end_frame:
                curr_seg_data = np.mean(curr_data[start_frame: end_frame, :], axis = 0)
                curr_seg_feats.append(curr_seg_data)
            else:
                curr_seg_data = np.mean(curr_data[start_frame: end_frame + 1, :], axis = 0)
                curr_seg_feats.append(curr_seg_data)
            
        curr_seg_feats = np.stack(curr_seg_feats, axis = 0)
        npaa.append(curr_seg_feats)
        print(len(curr_seg_feats), file=fw)
