import argparse
import numpy as np
import os
from npy_append_array import NpyAppendArray
import math
import torch.nn.functional as F
import json
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--which_set', default='train')
parser.add_argument('--data_path', default='/nobackup/users/junruin2/data/hubert/libri-train-clean-top-4096/feat_hubert_large_l21_ori_feats_no_sil_gt_bnd/top_4096/km_dir_4096')
parser.add_argument('--boundary_path', default='/nobackup/users/junruin2/LibriSpeech_vad')
args = parser.parse_args()

data_path = args.data_path
which_set = args.which_set 
vghubert_file = os.path.join(args.boundary_path, f'ls_train_clean_vad_syllable_{which_set}_oov_2048.json')
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

with open(vghubert_file, 'r') as fb:
    vghubert_feat_dict = json.load(fb)

seg_feats_path = os.path.join(data_path, 'seg_feats')

os.makedirs(seg_feats_path, exist_ok=True)
if os.path.exists(os.path.join(seg_feats_path, f'{which_set}_0_1.npy')):
    os.remove(os.path.join(seg_feats_path, f'{which_set}_0_1.npy'))
npaa = NpyAppendArray(os.path.join(seg_feats_path, f'{which_set}_0_1.npy'))


with open(os.path.join(seg_feats_path, f'{which_set}_0_1.len'), 'w') as fw:
    for idx, file_key in enumerate(filenames):

        curr_data = data[offsets[idx]: offsets[idx] + data_lens[idx]]
        curr_seg_feats = []
        for boundary_start, boundary_end, w in vghubert_feat_dict[file_key]:
            rounded_start = int(boundary_start // 320)
            rounded_end = int(boundary_end // 320)

            curr_seg_data = np.mean(curr_data[rounded_start: rounded_end, :], axis = 0)
            curr_seg_feats.append(curr_seg_data)

        curr_seg_feats = np.stack(curr_seg_feats, axis = 0)
        npaa.append(curr_seg_feats)
        print(len(curr_seg_feats), file=fw)
