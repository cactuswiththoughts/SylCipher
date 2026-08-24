import torch
import argparse
import numpy as np
import time
from tqdm import tqdm
import eval_segmentation
import data_loader
import os
import pickle


parser = argparse.ArgumentParser()
#parser.add_argument('--tsv_path', type=str, default='/nobackup/users/junruin2/LibriSpeech_vad/tsvs/valid.tsv')
#parser.add_argument('--boundary_json', type=str, default='/nobackup/users/junruin2/LibriSpeech_vad/ls_train_clean_vad_gradseg_style_valid.json')
#parser.add_argument('--seg_path', type=str, default='/nobackup/users/junruin2/data/hubert/libri-train-clean-vad/feat_hubert_large_l21_ori_feats_gradseg_wav2bnd/top_2048/km_dir_2048/discrete_speech_valid_clsattnbndkmeans/data_dict_bnd.pkl')
# Final result: 48.13378527059497 46.99373795761079 47.556930216460344 -2.368497109826584 55.56727059468205
#parser.add_argument('--tsv_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/tsvs/valid.tsv')
#parser.add_argument('--boundary_json', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/valid_vad_syllable.json')
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/data/LibriSpeech_vad/syllablelm_bnd/valid_data_bnd_dict.pkl')
# Final result (50ms): 70.42515039162483 82.8069522316761 76.11580033883969 17.581505713793643 75.40991034490172
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/data/LibriSpeech_vad/sylber_bnd/valid_data_bnd_dict.pkl')
# Final result (50ms): 70.42515039162483 82.8069522316761 76.11580033883969 17.581505713793643 75.40991034490172
#parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/data/LibriSpeech_vad/sylber_ft_bnd/valid_data_bnd_dict.pkl')

parser.add_argument('--tsv_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/val2.tsv')
parser.add_argument('--boundary_json', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/val2_vad_syllable.json')
# Final result (50ms):
#parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/syllablelm_bnd/val2_data_bnd_dict.pkl')
# Final result (50ms): 52.27472334445811 54.88243430152144 53.54684899464712 4.988473951129557 59.58858675986851
#parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/sylber_bnd/val2_data_bnd_dict.pkl')
#parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/sylber_ft_bnd/val2_data_bnd_dict.pkl')
parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/jstti_e2e_bnd/val2_data_bnd_dict.pkl')
#parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/wav2boundaries/dataset/spokencoco/train_wav2boundaries_sylber_bnd_2048_oov_e2e_refinement/val2_utts.pkl')
args = parser.parse_args()

frames_per_embedding = 160 #(not 320), instead of multiplying by 2 later
print("frame duration (s): %f" % (frames_per_embedding/16000), flush=True)

wavs, bounds, fns = data_loader.get_data_buckeye(args.tsv_path, args.boundary_json, load_wav=False)
with open(args.seg_path, 'rb') as f:
    pred_bound_dict = pickle.load(f)

ref_bounds = []
seg_bounds = []
for idx in tqdm(list(range(len(wavs)))):
    if (len(bounds[idx]) > 0) and (fns[idx] in pred_bound_dict):
        pred_bound = pred_bound_dict[fns[idx]]['seg_bound']
        audio_len = pred_bound.shape[-1]
        pred_boundaries = np.nonzero(pred_bound)[0][1:]

        boundaries = np.array(data_loader.get_bounds(bounds[idx])) // frames_per_embedding
        boundaries = np.minimum(boundaries[1:], audio_len-1)

        print('idx, boundaries, pred_boundaries:', idx, boundaries, pred_boundaries)  # XXX

        ref_bound = np.zeros(audio_len)
        seg_bound = np.zeros(audio_len)
        ref_bound[boundaries] = 1
        seg_bound[pred_boundaries] = 1
        ref_bound[-1] = 1
        seg_bound[-1] = 1
        
        ref_bounds.append(ref_bound)
        seg_bounds.append(seg_bound)

precision, recall, f = eval_segmentation.score_boundaries(ref_bounds, seg_bounds, 5)
os = eval_segmentation.get_os(precision, recall)*100
r_val = eval_segmentation.get_rvalue(precision, recall)*100
print("Final result:", precision*100, recall*100, f*100, os, r_val)
