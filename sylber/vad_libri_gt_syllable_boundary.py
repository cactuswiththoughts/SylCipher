import argparse
import torch
import json
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boundary_json', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/ls_train_clean_vad_syllable_train_oov_2048.json')
#    parser.add_argument('--boundary_json', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/ls_train_clean_vad_syllable_valid_oov_2048.json')
#    parser.add_argument('--boundary_json', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/ls_train_clean_vad_syllable_valid_true_oov_2048.json')
    parser.add_argument('--vads', default='/data/sls/scratch/limingw/data/LibriSpeech_vad/train-clean-100.json,/data/sls/scratch/limingw/data/LibriSpeech_vad/train-clean-360.json')
#    parser.add_argument('--output_path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/valid_true_vad_syllable.json')
    parser.add_argument('--output_path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/train_vad_syllable.json')
    return parser

def remove_silence(bounds, intervals, sr=16000):
    x = -torch.ones(bounds[-1][1])
    tokens = []
    for i, b in enumerate(bounds):
        x[b[0]:b[1]] = i
        tokens.append(b[2])

    new_x = []
    for itvl in intervals:
        s, e = itvl['start'], itvl['end']
        new_x.append(x[s:e])
    if len(new_x):
        new_x = torch.cat(new_x)
    else:
        new_x = -torch.ones(16000)

    units, counts = new_x.unique_consecutive(return_counts=True)
    units = units.long().tolist()
    counts = counts.long().tolist()

    new_bounds = []
    offset = 0
    for u, c in zip(units, counts):
        if u >= 0:
            new_bounds.append([offset, offset+c, tokens[u]])
        offset += c
    return new_bounds

def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.boundary_json, 'r') as f:
        bound_dict = json.load(f)

    vad_dict = {}
    for vad in args.vads.split(','):
        with open(vad, 'r') as f:
            vad_dict.update(json.load(f))

    new_bound_dict = {}
    for k in tqdm(bound_dict):
        new_bound_dict[k] = remove_silence(bound_dict[k], vad_dict[k])

    with open(args.output_path, 'w') as f_out:
        json.dump(new_bound_dict, f_out, indent=2)

if __name__ == '__main__':
    main()
