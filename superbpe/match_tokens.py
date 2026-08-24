import argparse
from copy import deepcopy
import json
import editdistance
from pathlib import Path
from tqdm import tqdm
SP = 'Ġ'

# A slightly cheating approach to create a dictionary of BPE tokens
# by matching with pyphen-generated tokens using edit distance
parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, required=True, help='Trained SuperBPE tokenizer directory (vocab.json + tokenizer.json)')
parser.add_argument('--dict_path', type=str, required=True, help='pyphen+ syllable dictionary to match against')
parser.add_argument('--out_dir', type=str, required=True, help='Output directory for the matched BPE+ vocab')
parser.add_argument('--lowercase', action='store_true')
args = parser.parse_args()
in_dir = Path(args.in_dir)
dict_path = args.dict_path
out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True, parents=True)

vocab = []
count = []
with open(dict_path, 'r') as f:
    lines = f.read().strip().split('\n')
    for l in lines:
        vocab.append(l.split()[0])
        count.append(l.split()[1])

with open(in_dir / 'vocab.json', 'r') as f:
    bpe_vocab = sorted(json.load(f))

with open(in_dir / 'tokenizer.json', 'r') as f:
    tokenizer_dict = json.load(f)

new_vocab = []
new_tokenizer_dict = deepcopy(tokenizer_dict)
new_tokenizer_dict['vocab'] = {}
for w_idx, w in tqdm(enumerate(vocab)):
    if args.lowercase:
        w = w.lower()
    else:
        w = w.upper()

    if w in bpe_vocab:
        print(f'{w} found in bpe_vocab', flush=True)
        new_vocab.append(w)
        new_tokenizer_dict['vocab'][w] = w_idx
    elif f'{SP}{w}' in bpe_vocab:
        print(f'{SP}{w} found in bpe_vocab', flush=True)
        new_tokenizer_dict['vocab'][f'{SP}{w}'] = w_idx 
        new_vocab.append(f'{SP}{w}')
    else:
        dist = [editdistance.eval(w, bpe.lstrip(SP)) for bpe in bpe_vocab]
        idx = sorted(range(len(dist)), key=lambda x:dist[x])[0]
        new_tokenizer_dict['vocab'][bpe_vocab[idx]] = w_idx 
        new_vocab.append(bpe_vocab[idx])

        print(f'{w}->{bpe_vocab[idx].lstrip(SP)}', flush=True)

#with open(out_dir / 'syllable_bpe_17k_2048.txt', 'w') as f:
with open(out_dir / 'syllable_bpe.txt', 'w') as f:
    for new_w, c in zip(new_vocab, count):
        print(f'{new_w} {c}', file=f)
