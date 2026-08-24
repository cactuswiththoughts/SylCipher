import argparse
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--in_path', required=True)
parser.add_argument('--out_path', required=True)
args = parser.parse_args()

clusts = []
with open(args.in_path, 'r') as f,\
    open(args.out_path, 'w') as f_out:
    lines = f.read().strip().split('\n')
    for l in tqdm(lines):
        clust = list(map(int, l.strip().split()))
        clusts.extend(clust)

np.save(args.out_path, np.asarray(clusts))


