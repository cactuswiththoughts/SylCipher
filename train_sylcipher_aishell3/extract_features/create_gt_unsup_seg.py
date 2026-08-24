import json
import os

## convert boundary format
import pickle
import numpy as np


for which_set in ['train', 'test']:
    read_gt_file = f'/nobackup/users/junruin2/data/hubert/aishell_3_vad/{which_set}_gradseg_style_forced_align.json'
    write_gt_file = f'/nobackup/users/limingw/data/hubert/aishell_3_vad/forced_align_bnd/{which_set}_data_bnd_dict.pkl'
    os.makedirs(os.path.dirname(write_gt_file), exist_ok = True)
        
    with open(read_gt_file, 'r') as f:
        gt_bnd_dict = json.load(f)
        
    new_gt_bnd_dict = {}
        
    file_keys = list(gt_bnd_dict.keys())

    for file_key in file_keys:
        if len(gt_bnd_dict[file_key]) > 0: 
            audio_len = int(gt_bnd_dict[file_key][-1][1] // 160) + 1
            bound = np.zeros(audio_len)
            for boundary_start, boundary_end, w in gt_bnd_dict[file_key]:
                rounded_start = int(boundary_start // 160)
                rounded_end = int(boundary_end // 160)
                bound[rounded_start] = 1
                bound[rounded_end] = 1

            item_dict = {}
            item_dict['ref_bound'] = bound
            item_dict['seg_bound'] = bound
        else:
            item_dict['ref_bound'] = []
            item_dict['seg_bound'] = []
            print(file_key, gt_bnd_dict[file_key])  # XXX
        
        new_gt_bnd_dict[file_key] = item_dict

    with open(write_gt_file, 'wb') as fwb:
        pickle.dump(new_gt_bnd_dict, fwb)
