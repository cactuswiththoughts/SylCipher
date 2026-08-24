import ast
import argparse
from copy import deepcopy
import pandas as pd
import json
import re
from pathlib import Path
import torchaudio
from tqdm import tqdm
PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
VOWEL_MAP = {'ō':'o', 'á':'a', 'ó':'o', 'à':'a', 'í':'i', 'é':'e', 'ī':'i', 'â':'a', 'ô':'o', 'ē':'e', 'ò':'o', 'ê':'e', 'ú':'u', 'ū':'u', 'ì':'i', 'ā':'a', 'î':'i', 'è':'e', 'ù':'u', 'ń':'n', 'û':'u', 'ő':'o', 'ǹ':'n', 'ű':'u', 'ḿ':'m', 'ı':'i', 'έ':'ε', 'ώ':'ω'}
VOWELS = 'aeiouεω'


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

parser = argparse.ArgumentParser()
parser.add_argument('--audio-dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mboshi_vad/wavs')
parser.add_argument('--text-path', default='/data/sls/scratch/limingw/data/mboshi-french-parallel-corpus/full_corpus_newsplit')
parser.add_argument('--tsv-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mboshi_vad/train.tsv')
parser.add_argument('--out-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mboshi_vad/train_gradseg_style.json')
parser.add_argument('--bad-path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mboshi_vad/train_bad_files.txt')
parser.add_argument('--no-tone', type=ast.literal_eval, default='False')
args = parser.parse_args()
Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
text_path = Path(args.text_path)

meta_dict = {}
vocab = []
with open(args.tsv_path, 'r') as f,\
    open(args.bad_path, 'w') as f_bad:
    lines = f.read().strip().split('\n')
    root = Path(lines.pop(0))
    for l in tqdm(lines):
        fn = l.strip().split('\t')[0]
        fpath = root / fn
        wav, sr = torchaudio.load(fpath)
        dur = wav.shape[-1]

        text_fn = text_path / (Path(fn).stem+'.mb.cleaned')
        with open(text_fn, 'r') as f:
            text = f.read().split('\n')[0].replace(' ', '')
 
        new_text = []
        for w in text.split():
            syls = syllabify(w)
            print(w, syls)
            for syl in syls:
                if syl not in vocab:
                    vocab.append(syl)
            new_text.extend(syls)
        text = deepcopy(new_text)
        if not len(text):
            print(f'{fn} without transcripts after postprocessing')
            print(fn, file=f_bad)
            continue
        else:
            print(fn, text)

        time_step = dur // len(text)

        meta_dict[fn] = [
            [i*time_step, (i+1)*time_step, w]
            for i, w in enumerate(text)
        ]

with Path(args.out_path).open('w', encoding='utf-8', newline='\n') as f_out:
    json.dump(meta_dict, f_out, ensure_ascii=False, indent=2)

print('Vocab size:', len(vocab))
