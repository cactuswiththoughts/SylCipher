import argparse
import os
import pandas as pd
import random
import torchaudio
from pathlib import Path
random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--wav_root', default='/data/sls/scratch/limingw/data/MDCC/dataset')
parser.add_argument('--in_path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mdcc/cnt_asr_train_metadata.csv')
parser.add_argument('--out_path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mdcc/train.tsv')
args = parser.parse_args()
wav_root = args.wav_root
out_path = Path(args.out_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
count = 0
outs = []

df = pd.read_csv(args.in_path)

with open(args.out_path, 'w') as f_out:
    print(args.wav_root, file=f_out)
    for wav_path, dur in zip(df['audio_path'], df['duration']):
        nframe = int(dur * 16e3)
        print(f'{wav_path}\t{nframe}', file=f_out)
        count += 1 
    print(f'Found {count} files')
