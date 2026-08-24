import numpy as np
import shutil
import os
import json
from tqdm import tqdm
from collections import defaultdict
NONWORD = ('<unk>')

topk = 2048
in_path = f'/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/lm_syllable_pyphen_plus.txt.txt'
out_path = f'/data/sls/scratch/limingw/data/LibriSpeech_vad/lm_syllable_pyphen_plus_oov_{topk}.txt'
dict_path = f'/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/syllable_pyphen_plus_dict_2048.txt'
with open(dict_path, 'r') as f:
    lines = f.readlines()

words = {}
for line in lines:
    splitted = line.strip().split()
    words[splitted[0]] = int(splitted[1])

top_words = sorted(words, key=lambda x:words[x], reverse=True) 

words['<OOV>'] = 0

with open(in_path, 'r') as f:
    in_sents = f.read().strip().split('\n')

with open(in_path, 'r') as fr,\
    open(out_path, 'w') as fw:
    for sent in tqdm(in_sents):
        to_write_list = []
        for c in sent.strip().split():
            if c in top_words:
                to_write_list.append(c)
            else:
                to_write_list.append('<OOV>')
                words['<OOV>'] += 1

        fw.write(" ".join(to_write_list) + '\n')

print(f'Replace {words["<OOV>"]} of {sum(words.values())} syllables with <OOV>')
