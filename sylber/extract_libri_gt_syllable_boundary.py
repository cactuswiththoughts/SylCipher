import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from textgrid import TextGrid
from tqdm import tqdm
from syllabify.syllabify import syllabify
NONWORD = ("<eps>", "sil", "spn")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', required=True, help='Utterance file list (one wav path per line)')
    parser.add_argument('--align_dir', required=True, help='MFA TextGrid alignments for LibriSpeech')
    parser.add_argument('--output_prefix', required=True, help='Prefix for the GradSeg-style JSON / pickle outputs')
    return parser

def read_alignment(align_path):
    tg = TextGrid.fromFile(align_path) 
    word_tier = next(t for t in tg.tiers if t.name.lower().startswith('word'))
    phone_tier = next(t for t in tg.tiers if t.name.lower().startswith('phone'))
    return word_tier, phone_tier        

def extract_syllable_boundaries(align_path, sr=16000):
    word_tier, phone_tier = read_alignment(align_path)
    out = []
    for w in word_tier.intervals:
        word = w.mark.strip()
        if (not word) or (word.lower() in NONWORD):
            continue

        # collect phonemes within this word's time span
        phones = [
            (p.mark, p.minTime, p.maxTime)
            for p in phone_tier.intervals
                if p.mark.strip() and p.minTime >= w.minTime and p.maxTime <= w.maxTime and (p.mark.strip().lower() not in NONWORD)
        ]
        if not phones:
            continue

        phns = [ph[0] for ph in phones]
        try:
            sylls = syllabify(phns)
        except:
            print(f'Failed to syllabify {phns}')
            continue

        idx = 0
        for syl in sylls:
            syl_phn_labels = [''.join([c for c in phn if c.isalpha()]) for syl_phns in syl for phn in syl_phns]
            syl_phones = phones[idx:idx+len(syl_phn_labels)]
            syl_start = int(syl_phones[0][1] * sr)
            syl_end = int(syl_phones[-1][2] * sr)
            out.append((syl_start, syl_end, '-'.join(syl_phn_labels)))
            idx += len(syl_phones)
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
    print('Number of syllable types without stress: ', len([''.join([c for c in syl if c.isalpha()]) for syl in syl_count]))

if __name__ == '__main__':
    main()
