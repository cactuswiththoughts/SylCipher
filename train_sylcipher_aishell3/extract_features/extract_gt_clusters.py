from pathlib import Path
from tqdm import tqdm

in_path = '/nobackup/users/limingw/data/hubert/aishell_vad_pinyin_no_tone/feat_hubert_large_ll60k_L21_forced_align_bnd/top_2048'
out_path = '/nobackup/users/limingw/data/hubert/aishell_vad_pinyin_no_tone/feat_hubert_large_ll60k_L21_forced_align_bnd/top_2048/km_dir_2048/seg_feats/km_idx_clsattnbndkmeans'
Path(out_path).mkdir(parents=True, exist_ok=True)

vocab = []
for x in ['train', 'dev', 'test']:
    with open(Path(in_path) / f'{x}_words.txt', 'r') as f,\
        open(Path(out_path) / f'{x}.km', 'w') as f_out:
        lines = f.read().strip().split('\n')
        for l in tqdm(lines):
            sent = []
            for w in l.strip().split():
                if not w in vocab:
                    vocab.append(w)
                sent.append(str(vocab.index(w)))
            print(' '.join(sent), file=f_out)

print(f'vocab size: {len(vocab)}')  

