import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from textgrid import TextGrid
from tqdm import tqdm
import pyphen
NONWORD = ("<eps>", "sil", "spn", "<unk>")
dic = pyphen.Pyphen(lang='en_US')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', required=True, help='Utterance file list (one wav path per line)')
    parser.add_argument('--align_dir', required=True, help='MFA TextGrid alignments for LibriSpeech')
    parser.add_argument('--output_prefix', required=True, help='Prefix for the GradSeg-style JSON / pickle outputs')
    return parser

def read_alignment(align_path):
    tg = TextGrid.fromFile(align_path)
    word_tier = next(t for t in tg.tiers if t.name.lower().startswith('word'))
    return word_tier

def extract_syllable_boundaries(align_path, sr=16000):
    word_tier = read_alignment(align_path)
    out = []
    for w in word_tier.intervals:
        word = w.mark.strip()
        dur = w.maxTime - w.minTime
        if (not word) or (word.lower() in NONWORD):
            continue
        syls = dic.inserted(word.lower()).split('-')
        
        for i, syl in enumerate(syls):
            syl_start = w.minTime+dur/len(syls)*i
            syl_start = int(syl_start * sr)
            syl_end = w.minTime+dur/len(syls)*(i+1)
            syl_end = int(syl_end * sr)
            out.append((syl_start, syl_end, syl))
    return out

def main():
    parser = get_parser()
    args = parser.parse_args()
    align_dir = Path(args.align_dir)
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    data_dict = {}
    syl_count = defaultdict(int)
    with open(args.filelist, 'r') as f,\
        open(args.output_prefix+'.txt', 'w') as f_out:
        fns = f.read().strip().split('\n')
        for i, fn in tqdm(list(enumerate(fns))):
#            if i > 10:  # XXX
#                break
            align_path = align_dir / fn.replace('.flac', '.TextGrid')
            syl_bound = extract_syllable_boundaries(align_path)
            text = []
            for _, _, syl_label in syl_bound:
                syl_count[syl_label] += 1
                text.append(syl_label)
            data_dict[fn] = syl_bound
            print(' '.join(text), file=f_out)

    with open(args.output_prefix+'.json', 'w') as f:
        json.dump(data_dict, f, indent=2)

    print('Number of syllable types: ', len(syl_count))
    print('number of syllable types without stress: ', len([''.join([c for c in syl if c.isalpha()]) for syl in syl_count]))

if __name__ == '__main__':
    main()
