import argparse
import ast
import json
import ToJyutping
from pathlib import Path
from tqdm import tqdm
import torchaudio
PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

def normalize(text):
    text = ''.join([c for c in text if c not in PUNCT and c != " "])
    return text

parser = argparse.ArgumentParser()
parser.add_argument('--tsv-path', required=True)
parser.add_argument('--text-path', required=True, help='MDCC transcription directory')
parser.add_argument('--out-path', required=True)
parser.add_argument('--bad-path', required=True)
parser.add_argument('--no-tone', type=ast.literal_eval, default='True')
args = parser.parse_args()
text_path = Path(args.text_path)

with open(args.tsv_path, 'r') as f_tsv:
    lines = f_tsv.read().strip().split('\n')
    root = lines.pop(0)

vocab = []
meta_dict = {}
with open(args.bad_path, 'w') as f_bad:
    for l in tqdm(lines):
        fn = l.strip().split('\t')[0]
        fpath = Path(root) / fn
        wav, sr = torchaudio.load(fpath)
        dur = wav.shape[-1]

        wav_id = Path(fn).stem
        text_fn = text_path / (wav_id + '.txt')
        with open(text_fn, 'r') as f:
            raw_text = f.read().strip()
            raw_text = normalize(raw_text)
            print(fn, raw_text, dur)

            # Detect non-Cantonese words
            syls = ToJyutping.get_jyutping_list(raw_text)

            text = []
            for c, syl in syls:
                if syl is None:
                    print(f'{fn}: {syls} contains non-Cantonese words, skip')
                    text = []
                    break
                else:
                    text.append(c)
                    if c not in vocab:
                        vocab.append(c)

            if not len(text):
                print(f'{fn} without transcripts after postprocessing')
                print(fn, file=f_bad)
                continue
            else:
                print(fn, text, dur)

            time_step = dur // len(text)

            meta_dict[fn] = [
                [i*time_step, (i+1)*time_step, w]
                for i, w in enumerate(text)
            ]

with Path(args.out_path).open('w', encoding='utf-8', newline='\n') as f_out:
    json.dump(meta_dict, f_out, ensure_ascii=False, indent=2)

print('Vocab size:', len(vocab))
