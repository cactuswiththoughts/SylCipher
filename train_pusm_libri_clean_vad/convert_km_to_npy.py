import argparse
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default='/nobackup/users/limingw/data/hubert/libri-train-clean-vad/feat_hubert_large_l21_ori_feats_sylber_bnd/top_2048/km_dir_2048/seg_feats/km_idx_clsattnbndkmeans/train.km')
parser.add_argument('--out_path', default='/nobackup/users/junruin2/data/hubert/libri-train-clean-vad/feat_hubert_large_l21_ori_feats_sylber_bnd/top_2048/km_dir_2048/train.npy')
args = parser.parse_args()

clusts = []
with open(args.in_path, 'r') as f,\
    open(args.out_path, 'w') as f_out:
    lines = f.read().strip().split('\n')
    for l in tqdm(lines):
        clust = list(map(int, l.strip().split()))
        clusts.extend(clust)

np.save(args.out_path, np.asarray(clusts))


