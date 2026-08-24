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
    parser.add_argument('--tokenizer_path', required=True, help='Matched BPE+ vocab file (syllable_bpe.txt)')
    parser.add_argument('--output_prefix', required=True, help='Prefix for tokenized JSON / text outputs')
    return parser

class GreedyTokenizer:
    def __init__(self, vocab):
        self.vocab = sorted(vocab, key=lambda x:len(x), reverse=True)
        print(self.vocab[:10])
        self.word2token = {}
        self.oov = '<OOV>'

    @staticmethod
    def from_file(data_json):
        with open(data_json, 'r') as f:
            vocab = [l.strip().split()[0] for l in f.read().strip().split('\n')]
        return GreedyTokenizer(vocab)

    def tokenize(self, w):
        if not w in self.word2token:
            self.word2token[w] = []
        
        offset = 0
        tokens = []
        for token in self.vocab:
            if w[offset:].startswith(token):
                tokens.append(token)
                offset += len(token)
                if offset >= len(w):
                    self.word2token[w].extend(tokens)
                    return tokens
        tokens.append(self.oov)
        self.word2token[w].extend(tokens)
        return tokens

def main():
    parser = get_parser()
    args = parser.parse_args()
    metadata_path = Path(args.metadata_path)
    with open(metadata_path, 'r') as f:
        data_dict = json.load(f) 

    tokenizer = GreedyTokenizer.from_file(args.tokenizer_path)

    Path(args.output_prefix+'.json').parent.mkdir(parents=True, exist_ok=True)
    new_data_dict = {}
    vocab = []
    for fn, word_list in tqdm.tqdm(list(data_dict.items())):
        new_data_dict[fn] = []
        for start, end, w in word_list:
            dur = end - start
            tokens = tokenizer.tokenize(w.lower())
            n_p = len(tokens)
            for i, p in enumerate(tokens):
                new_data_dict[fn].append([start+i*(dur//n_p), start+(i+1)*(dur//n_p), p.lstrip(SP)])
                if not w in vocab:
                    print(f'{w.lower()}->{tokens}', flush=True)
                    vocab.append(w)
    
    with open(args.output_prefix+'.json', 'w') as f:
        json.dump(new_data_dict, f, indent=2)

if __name__ == '__main__':
    main() 
