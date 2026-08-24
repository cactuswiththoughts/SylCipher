import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import pyphen
import syllapy
import wordsegment
NONWORD = ("<eps>", "sil", "spn", "<unk>")
VOWELS = 'aeiouy'
dic = pyphen.Pyphen(lang='en_US')
wordsegment.load()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Unpaired LM text (one sentence per line)')
    parser.add_argument('--output_prefix', required=True, help='Prefix for syllabified text outputs')
    return parser

def has_vowel(seg):
    for v in VOWELS:
        if v in seg:
            return True
    return False

def merge_novowel_segments(segs):
    new_segs = []
    n_seg = len(segs)
    i = 0
    while i < n_seg:
        seg = segs[i]
        if not has_vowel(seg):
            # If seg does not contain vowels and is at the end, add it to the previous seg
            if i == n_seg - 1:
                if len(new_segs):
                    new_segs[-1] += seg
                else:
                    new_segs = [seg]
            # Otherwise, add it to the next seg
            else:
                new_segs.append(seg+segs[i+1])
                i += 1
        i += 1
    return new_segs

def naive_syllabify(w):
    w0 = w
    # Drop a truly silent 'e' (unless -le ending)    
    if w.endswith('e') and not re.search(rf'[^ {VOWELS}]le$', w):
        w = w[:-1]    

    # For 'ed' endings pronounced as /t/ or /d/, strip the 'e'
    #     (i.e., drop the e in -ed unless the root ends in 't' or 'd')
    if w.endswith("ed") and len(w) > 2 and w[-3] not in ("t","d"):
        # e.g. 'jumped' → 'jumpd', 'baked' → 'bakd'
        w = w[:-2] + w[-1]

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

    if w0 != w:
        if w0.endswith('e'):
            w2 += 'e'
        elif w0.endswith('ed'):
            w2 = w2[:-1] + 'ed'
    return w2.split('-')

def extract_syllable_boundaries(sent, sr=16000, wrd2syl={}):
    out = []
    for word in sent.strip().split():
        if (not word) or (word.lower() in NONWORD):
            continue

        word = word.lower()
        if word not in wrd2syl:
            segs = wordsegment.segment(word)

            if len(segs) > 1:
                print(f'break {word} into {segs}', flush=True)
            
            syls = []
            for seg in segs:
                syls_ = dic.inserted(seg).split('-')
                if (len(syls_) == 1) and (syllapy.count(seg) > len(syls_)):
                    syls_ = naive_syllabify(seg)
                syls.extend(syls_)

            wrd2syl[word] = syls
            print(f'{word} -> {syls}')
        else:
            syls = wrd2syl[word]

        for i, syl in enumerate(syls):
            out.append(syl)
    return out, wrd2syl

def main():
    parser = get_parser()
    args = parser.parse_args()
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    wrd2syl = defaultdict(list)
    syl_count = defaultdict(int)
    with open(args.input_file, 'r') as f,\
        open(args.output_prefix+'.txt', 'w') as f_out:
        sents = f.read().strip().split('\n')
        for i, sent in tqdm(list(enumerate(sents))):
#            if i > 10:  # XXX
#                break
            syl_labels, wrd2syl = extract_syllable_boundaries(sent, wrd2syl=wrd2syl)
            text = []
            for syl_label in syl_labels:
                syl_count[syl_label] += 1
                text.append(syl_label)
            print(' '.join(text), file=f_out)

    print('Number of syllable types: ', len(syl_count))
    print('number of syllable types without stress: ', len([''.join([c for c in syl if c.isalpha()]) for syl in syl_count]))

if __name__ == '__main__':
    main()
