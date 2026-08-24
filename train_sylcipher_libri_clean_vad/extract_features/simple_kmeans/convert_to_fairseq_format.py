import argparse
import pickle as pkl
from tqdm import tqdm
from npy_append_array import NpyAppendArray

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default='/data/sls/scratch/limingw/data/LibriSpeech_vad/sylber_bnd/train_data_bnd_dict_segment_features.pkl')
parser.add_argument('--out_prefix', default='/data/sls/scratch/limingw/data/LibriSpeech_vad/sylber_bnd/train')
args = parser.parse_args()

with open(args.in_path, 'rb') as f:
    feat_dict = pkl.load(f)

with open(args.out_prefix+'.tsv', 'w') as f_tsv,\
    open(args.out_prefix+'_0_1.len', 'w') as f_len,\
    NpyAppendArray(args.out_prefix+'_0_1.npy', delete_if_exists=True) as npaa:
    for fn in tqdm(list(sorted(feat_dict))):
        print(fn, file=f_tsv)
        print(feat_dict[fn].shape, flush=True)
        print(feat_dict[fn].shape[0], file=f_len)
        npaa.append(feat_dict[fn])

