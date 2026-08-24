import argparse
import pickle
import numpy as np
import os
import torch
import math


parser = argparse.ArgumentParser()
parser.add_argument('--write_dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/wav2boundaries/spokencoco/dataset/train_wav2boundaries_sylber_2048_oov_e2e_refinement', help='output directory')
parser.add_argument('--raw_dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/wavs', help='raw data directory')
parser.add_argument('--data_dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/feat/hubert_large_ll60k_sylber_bnd/top_2048/km_dir_2048', help='manifest directory')
parser.add_argument('--in_boundary_dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/feat/hubert_large_ll60k_sylber_bnd/top_2048/km_dir_2048/model_codebook_0.3_clsattn_bnd_kmeans_encoder_only_diff_bnd')
parser.add_argument('--split', default='train')
args = parser.parse_args()

which_set = args.split
parent_dir = args.raw_dir
file_dir = args.data_dir
bnd_dir = args.write_dir
manifest_path = args.in_boundary_dir

file_dir = os.path.join(file_dir, f'discrete_speech_{which_set}_clsattnbndkmeans')
with open(os.path.join(file_dir, 'file_list.txt'), 'r') as f:
    all_file_lines = f.readlines()
print(len(all_file_lines))

with open(os.path.join(file_dir, 'bnd_errors.txt'), 'r') as f:
    bad_file_lines = f.readlines()

bad_files = set([f.strip().split()[0] for f in bad_file_lines])
print(bad_files)
selected_files = [f.strip() for f in all_file_lines if f.strip() not in bad_files]
print(len(selected_files), selected_files[:10])

os.makedirs(bnd_dir, exist_ok = True)
with open(os.path.join(manifest_path, f'predicted_boundaries_{which_set}.pkl'), 'rb') as fb:
    boundary_dict = pickle.load(fb)
    
with open(os.path.join(bnd_dir, which_set), 'w') as fw:

    for filename in selected_files:
        boundaries = boundary_dict[filename].numpy()
      
        spk = filename.split('/')[1]
      
        boundary_locs = list(np.where(boundaries == 1)[0])

        if boundary_locs[0] != 0:
            boundary_locs = [0] + boundary_locs
          
        fw.write(os.path.join(parent_dir, filename) + ' ' + spk + ' ')
        word_list = []
        for start, end in zip(boundary_locs[:-1], boundary_locs[1:]):
            start_in_sec = start * 0.02
            start_in_sec = math.floor(start_in_sec * 100)/100.0
            end_in_sec = end * 0.02
            end_in_sec = math.floor(end_in_sec * 100)/100.0
            word_list.append(str(start_in_sec) + "," + str(end_in_sec))
          
        fw.write(' '.join(word_list) + '\n')
    
    
###  python format_boundary.py /nobackup/users/junruin2/wav2boundaries/librispeech-clean-no-sil-top-2048/hubert_wav2boundaries_gradseg_unsup_diff_bnd_large_batch_from_init_codebook_0/train /nobackup/users/junruin2/wav2boundaries/librispeech-clean-no-sil-top-2048/dataset/train_hubert_wav2boundaries_gradseg_unsup_diff_bnd_large_batch_from_init_codebook_0
