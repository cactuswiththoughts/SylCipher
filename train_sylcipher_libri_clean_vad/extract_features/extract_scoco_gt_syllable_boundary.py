import argparse
import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from syllabify.syllabify import syllabify
NONWORD = ("<SPOKEN_NOISE>", "<UNK>", "<None>", 'SPN')
OOV = '<OOV>'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/test_file_list.txt')
    parser.add_argument('--align_path', default='/data/sls/scratch/sbhati/data/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy_with_alignments_new_adapt_syllable.json')
    parser.add_argument('--output_prefix', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/test_syllable')
    return parser

def read_alignment(ali):
    alignment = [[float(a.split('__')[0]), float(a.split('__')[2]), a.split('__')[1]] for a in ali.split() if len(a.split('__')) == 3]
    return alignment

def extract_syllable_boundaries(align_dict, sr=16000):
    word_ali = read_alignment(align_dict['text_alignment'])
    phn_ali = read_alignment(align_dict['phone_alignment'])
    out = []
    for start, end, word in word_ali:
        if word in NONWORD:
            continue

        phones = [
            (phn, s, e) for s, e, phn in phn_ali
            if phn and s >= start and e <= end and (phn not in NONWORD) 
        ]
        if not len(phones):
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
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    with open(args.align_path, 'r') as f:
        data = json.load(f)['data']
        wav2align = {d['caption']['wav']:d for d in data}
    print(len(wav2align))

    data_dict = {}
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
                syl_bound = extract_syllable_boundaries(align_dict)
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
