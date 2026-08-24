import ast
import argparse
from copy import deepcopy
import pandas as pd
import json
import torchaudio
from pathlib import Path
from tqdm import tqdm
PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
# ['h', 'i', 't', '-', 'l', 'ō', '.', 'e', 'g', 'u', 'á', ' ', 'k', 'ó', 'n', 'à', 'í', 'é', 's', 'ī', 'p', 'b', 'â', '?', 'ô', 'ē', 'a', '2', 'ò', '!', 'm', 'ê', 'ú', '̍', ',', 'ū', 'ì', '̄', 'ā', 'j', 'î', 'è', 'o', '。', 'ù', '̋', '̂', 'ń', 'v', 'f', 'c', 'w', 'r', 'x', 'û', 'd', 'y', '"', 'q', "'", 'ő', 'ㄦ', 'ˋ', 'z', 'ǹ', 'ű', '，', 'ㄎ', 'ㄜ', '〈', '〉', '《', '》', '(', ')', ';', ':', '3', '─', '‘', '’', '“', '”', 'ḿ', 'ı', '？', 'ㄌ', 'ㄧ', 'ㄢ', 'ˊ']
VOWEL_MAP = {'ō':'o', 'á':'a', 'ó':'o', 'à':'a', 'í':'i', 'é':'e', 'ī':'i', 'â':'a', 'ô':'o', 'ē':'e', 'ò':'o', 'ê':'e', 'ú':'u', 'ū':'u', 'ì':'i', 'ā':'a', 'î':'i', 'è':'e', 'ù':'u', 'ń':'n', 'û':'u', 'ő':'o', 'ǹ':'n', 'ű':'u', 'ḿ':'m', 'ı':'i'}

def syllabify(w):
    w = ''.join([' ' if c in PUNCT else c for c in w])
    syls = [x for x in w.split() if len(x)]
    return syls

def remove_tone(syls):
    new_syls = []
    for syl in syls:
        syl = ''.join([VOWEL_MAP.get(c, c) for c in syl if c.isalpha()])
        new_syls.append(syl)
    return new_syls

parser = argparse.ArgumentParser()
parser.add_argument('--audio-dir', required=True, help='Directory of SuiSiann wavs after VAD')
parser.add_argument('--text-path', required=True, help='SuiSiann.csv')
parser.add_argument('--tsv-path', required=True)
parser.add_argument('--out-path', required=True)
parser.add_argument('--bad-path', required=True)
parser.add_argument('--no-tone', type=ast.literal_eval, default='False')
args = parser.parse_args()
Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.text_path)

text_dict = {}
char_dict = {}
dur_dict = {}
for fn, sent_type, char, text, dur in tqdm(list(zip(df['音檔'], df['來源'], df['漢字'], df['羅馬字'], df['長短']))):
    wav_id = Path(fn).name
    char_dict[wav_id] = char.split()
    text_dict[wav_id] = text.lower().split()
    dur_dict[wav_id] = dur

meta_dict = {}
vocab = []
with open(args.tsv_path, 'r') as f,\
    open(args.bad_path, 'w') as f_bad:
    lines = f.read().strip().split('\n')
    root = Path(lines.pop(0))
    for l in tqdm(lines):
        fn = l.strip().split('\t')[0]
        wav_id = Path(fn).name
        fpath = root / fn
        if wav_id in text_dict:
            wav, sr = torchaudio.load(fpath)
            dur = wav.shape[-1]
            text = text_dict[wav_id]
            char = char_dict[wav_id]

            new_text = []
            for w in text:
                syls = syllabify(w)
                if args.no_tone:
                    syls = remove_tone(syls)

                for syl in syls:
                    if syl not in vocab:
                        vocab.append(syl)
                new_text.extend(syls)
            text = deepcopy(new_text)
            if not len(text):
                print(f'{fn} without transcripts after postprocessing')
                print(fn, file=f_bad)
                continue

            time_step = dur // len(text)

            meta_dict[fn] = [
                [i*time_step, (i+1)*time_step, w]
                for i, w in enumerate(text)
            ]
        else:
            print(f'{fn} without transcripts')
            print(fn, file=f_bad)

with Path(args.out_path).open('w', encoding='utf-8', newline='\n') as f_out:
    json.dump(meta_dict, f_out, ensure_ascii=False, indent=2)

print('Vocab size:', len(vocab))
