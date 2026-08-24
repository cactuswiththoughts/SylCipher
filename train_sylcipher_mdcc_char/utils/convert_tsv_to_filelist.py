import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--tsv', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/aishell_vad/train.tsv')
parser.add_argument('--filelist', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/aishell_vad/train_file_list.txt')
args = parser.parse_args()

with open(args.tsv, 'r') as f_tsv,\
    open(args.filelist, 'w') as f_list:
    lines = f_tsv.read().strip().split('\n')
    _ = lines.pop(0)
    for l in tqdm(lines):
        print(l.strip().split('\t')[0], file=f_list)
