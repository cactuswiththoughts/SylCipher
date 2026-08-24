import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from textgrid import TextGrid
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
    parser.add_argument('--filelist', required=True, help='Utterance file list (one wav path per line)')
    parser.add_argument('--align_dir', required=True, help='MFA TextGrid alignments for LibriSpeech')
    parser.add_argument('--output_prefix', required=True, help='Prefix for the GradSeg-style JSON / pickle outputs')
    return parser

def read_alignment(align_path):
    tg = TextGrid.fromFile(align_path)
    word_tier = next(t for t in tg.tiers if t.name.lower().startswith('word'))
    return word_tier

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

def extract_syllable_boundaries(align_path, sr=16000, wrd2syl={}):
    word_tier = read_alignment(align_path)
    out = []
    for w in word_tier.intervals:
        word = w.mark.strip()
        dur = w.maxTime - w.minTime
        if (not word) or (word.lower() in NONWORD):
            continue

        word = word.lower()
        if word not in wrd2syl:
            segs = wordsegment.segment(word)
# XXX            segs = merge_novowel_segments(segs)

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
            syl_start = w.minTime+dur/len(syls)*i
            syl_start = int(syl_start * sr)
            syl_end = w.minTime+dur/len(syls)*(i+1)
            syl_end = int(syl_end * sr)
            out.append((syl_start, syl_end, syl))
    return out, wrd2syl

def main():
    parser = get_parser()
    args = parser.parse_args()
    align_dir = Path(args.align_dir)
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    data_dict = {}
    wrd2syl = defaultdict(list)
    syl_count = defaultdict(int)
    with open(args.filelist, 'r') as f,\
        open(args.output_prefix+'.txt', 'w') as f_out:
        fns = f.read().strip().split('\n')
        for i, fn in tqdm(list(enumerate(fns))):
#            if i > 10:  # XXX
#                break
            align_path = align_dir / fn.replace('.flac', '.TextGrid')
            syl_bound, wrd2syl = extract_syllable_boundaries(align_path, wrd2syl=wrd2syl)
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
