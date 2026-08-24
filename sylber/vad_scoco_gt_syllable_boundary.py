import argparse
import json
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boundary_json', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/val2_syllable.json')
    parser.add_argument('--output_path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/val2_vad_syllable.json')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.boundary_json, 'r') as f:
        segment_dict = json.load(f)

    new_segment_dict = {}
    for fn in tqdm(segment_dict):
        segments = segment_dict[fn]
        offset = 0
        new_segments = []
        for seg in segments:
            dur = seg[1] - seg[0]
            new_segments.append([offset, offset+dur, seg[2]])
            offset += dur
        new_segment_dict[fn] = new_segments

    with open(args.output_path, 'w') as f:
        json.dump(new_segment_dict, f, indent=2)

if __name__ == '__main__':
    main()
