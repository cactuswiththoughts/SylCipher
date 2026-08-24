import ast
import argparse
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm
PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
# 'a', 'b', 'd', 'e', 'f', 'g', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'µ', 'ß', 'à', 'á', 'â', 'ä', 'æ', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü'
VOWEL_MAP = {'à':'a', 'á':'a', 'â':'a', 'ä':'a', 'è':'e', 'é':'e', 'ê':'e', 'ë':'e', 'ì':'i', 'í':'i', 'î':'i', 'ï':'i', 'ò':'o', 'ó':'o', 'ô':'o', 'õ':'o', 'ö':'o', 'ù':'u', 'ú':'u', 'û':'u', 'ü':'u'}
VOWELS = 'aeiouæ'


def syllabify(w):
    w = ''.join([VOWEL_MAP.get(c, c) for c in w if c not in PUNCT])
    # For 3+ consonants between vowels, split V C-CC… V → VC-CC…V
    w = re.sub(
        rf"(?<=[{VOWELS}][^{VOWELS}])"      # look-behind: V + C
        rf"(?=[^ {VOWELS}]"                # look-ahead start: C
        rf"{{2,}}[{VOWELS}])",             # at least 2 more C’s, then V
        "-",
        w
    )
    # VCCV -> VC | CV
    w1 = re.sub(rf'(?<=[{VOWELS}][^{VOWELS}])(?=[^{VOWELS}][{VOWELS}])', '-', w.lower())
    # VCV -> V | CV   
    w2 = re.sub(rf'(?<=[{VOWELS}])(?=[^{VOWELS}][{VOWELS}])', '-', w1)
    return w2.split('-')

def vad(ali, text, intervals):
    dur = max(ali)*160
    boundaries = np.zeros(dur+1)
    boundaries[0] = 1

    phn2wrd = [i for i, w in enumerate(text) for _ in w]
    print('boundary, voice intervals:', [a*160 for a in ali], intervals)  # XXX
    for i, (start, end) in enumerate(zip([0]+ali[:-1], ali)):
        boundaries[start*160:end*160] = phn2wrd[i]+1

    new_boundaries = []
    for itvl in intervals:
        new_boundaries.append(boundaries[itvl[0]:itvl[1]])
    boundaries = np.concatenate(new_boundaries)
    units, counts = torch.unique_consecutive(torch.from_numpy(boundaries), return_counts=True)
    print(text)  # XXX

    offset = 0
    segments = []
    for u, c in zip(units.tolist(), counts.tolist()):
        if u == 0:
            continue
        segments.append([offset, offset+c, text[int(u)-1]])
        offset += c
    print(segments)

    return segments

parser = argparse.ArgumentParser()
parser.add_argument('--audio-dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mboshi_vad/wavs')
parser.add_argument('--text-path', default='/data/sls/scratch/limingw/data/mboshi-french-parallel-corpus/forced_alignments_supervised_spkr/limsi-align/trn0.seg')
parser.add_argument('--tsv-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mboshi_vad/train.tsv')
parser.add_argument('--vad-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mboshi/train.vads')
parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mboshi_vad/train_gradseg_style.json')
parser.add_argument('--bad-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mboshi_vad/train_bad_files.txt')
parser.add_argument('--no-tone', type=ast.literal_eval, default='False')
args = parser.parse_args()
Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
text_path = Path(args.text_path)

text_dict = defaultdict(str)
align_dict = defaultdict(list)
with open(text_path, 'r', encoding='latin1') as f:
    lines = f.read().strip().split('\n')
    for l in tqdm(lines):
        if l.startswith('#'):
            if l.startswith('#@ sid'):
                fn = l.split('=')[-1].split('.')[0]+'.wav'
            elif l.startswith('#@ word'):
                text_dict[fn] += ' '
            continue

        fn = l.split()[0]+'.wav'
        w = l.split(':')[1].split()[0][1]
        start = int(l.split()[3])

        if (w == '.') or (w in PUNCT):
            continue
        else:
            text_dict[fn] += w.lower()
            align_dict[fn].append(start)

meta_dict = {}
vocab = []
with open(args.tsv_path, 'r') as f,\
    open(args.vad_path, 'r') as f_vad,\
    open(args.bad_path, 'w') as f_bad:
    lines = f.read().strip().split('\n')
    root = Path(lines.pop(0))

    vad_intervals = f_vad.read().strip().split('\n')

    for l, intervals in tqdm(zip(lines, vad_intervals)):
        intervals = [list(map(int, itvl.split(':'))) for itvl in intervals.split()]
        fn = l.strip().split('\t')[0]
        fpath = root / fn
        wav, sr = torchaudio.load(fpath)
        dur = wav.shape[-1]

        if fn in text_dict:
            text = text_dict[fn]
            ali = align_dict[fn]
        else:
            assert '_'.join(fn.split('_')[1:]) in text_dict
            text = text_dict['_'.join(fn.split('_')[1:])]
            ali = align_dict['_'.join(fn.split('_')[1:])]
        print(fn, text, ali)  # XXX

        new_text = []
        for w in text.strip().split():
            syls = syllabify(w)
            for syl in syls:
                if syl not in vocab:
                    vocab.append(syl)
            new_text.extend(syls)

        text = deepcopy(new_text)
        segments = vad(ali, text, intervals)
        if not len(text):
            print(f'{fn} without transcripts after postprocessing')
            print(fn, file=f_bad)
            continue
        else:
            print(fn, text)

        time_step = dur // len(text)

        meta_dict[fn] = segments

with Path(args.out_path).open('w', encoding='utf-8', newline='\n') as f_out:
    json.dump(meta_dict, f_out, ensure_ascii=False, indent=2)

print('Vocab size:', len(vocab))
