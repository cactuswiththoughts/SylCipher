import argparse
import json
from pathlib import Path
import tqdm
from tokenizers.models import BPE
from tokenizers import Tokenizer
SP = 'Ġ'
VOWELS = set("aeiouyAEIOUY")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', required=True, help='GradSeg-style metadata JSON')
    parser.add_argument('--tokenizer_path', required=True, help='SuperBPE tokenizer.json')
    parser.add_argument('--output_prefix', required=True, help='Prefix for tokenized JSON / text outputs')
    return parser

def has_vowel(piece):
    return any(ch in VOWELS for ch in piece)

def split_on_nonconsecutive_vowels(token):
    """
    Split a token into parts so that no part contains
    more than 1 non-consecutive vowels.
    """
    parts = []
    current = []
    vowel_positions = []

    for i, ch in enumerate(token):
        current.append(ch)
        if ch in VOWELS:
            vowel_positions.append(len(current) - 1)

        if len(vowel_positions) > 1:
            # break before this vowel
            cut_index = vowel_positions[-1]  # last vowel pos
            parts.append("".join(current[:cut_index]))
            current = current[cut_index:]
            vowel_positions = [v - cut_index for v in vowel_positions[-2:]]

    if current:
        parts.append("".join(current))

    return parts

def enforce_vowel_constraint(parts):
    """
    Merge consecutive pieces until each piece contains at least one vowel.
    """
    parts = [p for p in parts if len(p)]
    new_parts = []

    buffer = ""
    for i, p in enumerate(parts):
        buffer += p
        if has_vowel(buffer):
            if i < len(parts)-1:
                # e.g., MO O -> MOO
                if (p[-1] in VOWELS) and (parts[i+1][0] in VOWELS):
                    continue
                elif parts[i+1][0] == 'E':
                    continue
            new_parts.append(buffer)
            buffer = ""
    if buffer:  # leftover with no vowels
        # merge into the last piece if possible
        if new_parts:
            new_parts[-1] += buffer
        else:
            new_parts.append(buffer)
    return new_parts

def process_tokens(tokens):
    first_split = [part for token in tokens for part in split_on_nonconsecutive_vowels(token)]
    final_parts = enforce_vowel_constraint(first_split)
    return final_parts

def main():
    parser = get_parser()
    args = parser.parse_args()
    metadata_path = Path(args.metadata_path)
    with open(metadata_path, 'r') as f:
        data_dict = json.load(f) 

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    Path(args.output_prefix+'.json').parent.mkdir(parents=True, exist_ok=True)
    new_data_dict = {}
    vocab = []
    for fn, word_list in tqdm.tqdm(list(data_dict.items())):
        new_data_dict[fn] = []
        for start, end, w in word_list:
            dur = end - start
            tokens = tokenizer.encode(w).tokens
            tokens = process_tokens(tokens)
            n_p = len(tokens)
            for i, p in enumerate(tokens):
                if p == '<unk>':
                    p = '<OOV>'
                new_data_dict[fn].append([start+i*(dur//n_p), start+(i+1)*(dur//n_p), p.lstrip(SP)])
                if not w in vocab:
                    print(f'{w}->{tokens}', flush=True)
                    vocab.append(w)
    
    with open(args.output_prefix+'.json', 'w') as f:
        json.dump(new_data_dict, f, indent=2)

if __name__ == '__main__':
    main() 
