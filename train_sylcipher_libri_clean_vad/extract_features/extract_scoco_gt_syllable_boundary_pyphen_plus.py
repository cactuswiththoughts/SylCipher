import argparse
import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import pyphen
import syllapy
import wordsegment
NONWORD = ("<SPOKEN_NOISE>", "<UNK>", "<None>", 'SPN')
VOWELS = 'aeiouy'
OOV = '<OOV>'
dic = pyphen.Pyphen(lang='en_US')
wordsegment.load()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/test_file_list.txt')
    parser.add_argument('--align_path', default='/data/sls/scratch/sbhati/data/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy_with_alignments_new_adapt_syllable.json')
    parser.add_argument('--output_prefix', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/test_syllable_pyphen_plus')
    return parser

def read_alignment(ali):
    alignment = [[float(a.split('__')[0]), float(a.split('__')[2]), a.split('__')[1]] for a in ali.split() if len(a.split('__')) == 3]
    return alignment

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

def extract_syllable_boundaries(align_dict, wrd2syl, sr=16000):
    word_ali = read_alignment(align_dict['text_alignment'])
    out = []
    for start, end, word in word_ali:
        if word in NONWORD:
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

        dur = end - start
        for i, syl in enumerate(syls):
            syl_start = start + dur/len(syls)*i
            syl_start = int(syl_start * sr)
            syl_end = end + dur/len(syls)*(i+1)
            syl_end = int(syl_end * sr)
            out.append((syl_start, syl_end, syl))
    return out, wrd2syl

def main():
    parser = get_parser()
    args = parser.parse_args()
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    with open(args.align_path, 'r') as f:
        data = json.load(f)['data']
        wav2align = {d['caption']['wav']:d for d in data}
    print(len(wav2align))

    data_dict = {}
    wrd2syl = defaultdict(list)
    syl_count = defaultdict(int)
    with open(args.filelist, 'r') as f,\
        open(args.output_prefix+'.txt', 'w') as f_out,\
        open(args.output_prefix+'_bnd_errors.txt', 'w') as f_err:
        fns = f.read().strip().split('\n')
        for i, fn in tqdm(list(enumerate(fns))):
#            if i > 10:  # XXX
#                break
            fn = fn.strip()
            if fn in wav2align:
                align_dict = wav2align[fn]
                syl_bound, wrd2syl = extract_syllable_boundaries(align_dict, wrd2syl)
                text = []
                for _, _, syl_label in syl_bound:
                    syl_count[syl_label] += 1
                    text.append(syl_label)
                data_dict[fn] = syl_bound
                print(' '.join(text), file=f_out)
            else:
                print(OOV, file=f_out)
                print(fn, file=f_err)

    with open(args.output_prefix+'.json', 'w') as f:
        json.dump(data_dict, f, indent=2)

    print('Number of syllable types: ', len(syl_count))
    print('Number of syllable types without stress: ', len([''.join([c for c in syl if c.isalpha()]) for syl in syl_count]))

if __name__ == '__main__':
    main()
