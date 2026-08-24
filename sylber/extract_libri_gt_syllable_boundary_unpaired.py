import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from g2p_en import G2p
from tqdm import tqdm
from syllabify.syllabify import syllabify
NONWORD = ("<eps>", "sil", "spn", 'sp')
# Initialize G2P model
g2p = G2p()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', required=True, help='Unpaired LM text (one sentence per line)')
    parser.add_argument('--output_prefix', required=True, help='Prefix for syllabified text outputs')
    return parser

def syllabify_text(text, word2syl):
    syl_text = []
    for w in text:
        if w in word2syl:
            syl_text.extend(word2syl[w])
        else:
            phns = g2p(w)
            phns = [p for p in phns if re.match(r"[A-Z]+[0-2]?$", p)]
            if phns is not None:
                try:
                    sylls = syllabify(phns)
                except:
                    print(f'Failed to syllabify {phns}')
                    continue

                word2syl[w] = []
                for syl in sylls:
                    syl_phn_labels = [''.join([c for c in phn if c.isalpha()]) for syl_phns in syl for phn in syl_phns]
                    syl_text.append('-'.join(syl_phn_labels))
                    word2syl[w].append('-'.join(syl_phn_labels))
    return syl_text, word2syl

def main():
    parser = get_parser()
    args = parser.parse_args()

    word2syl = {}
    with open(args.text_path, 'r') as f,\
        open(args.output_prefix+'.txt', 'w') as f_out:
        lines = f.read().strip().split('\n')
        for i, l in tqdm(list(enumerate(lines))):
            words = l.strip().split()
            syls, word2syl = syllabify_text(words, word2syl)
            if i < 10:
                print(l, syls, flush=True) 
            print(' '.join(syls), file=f_out)

if __name__ == '__main__':
    main()
