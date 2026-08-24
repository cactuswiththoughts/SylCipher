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
#parser.add_argument('--boundary_json', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/ls_train_clean_vad_gradseg_style_valid.json')
parser.add_argument('--tsv_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/tsvs/valid_true.tsv')
parser.add_argument('--boundary_json', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/ls_train_clean_vad_gradseg_style_valid_true.json')

# Final result (50ms): 74.62269038162667 72.8557975534741 73.72865967722475 -2.3677688637551486 77.61656984100586
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/km_dir_top2048/discrete_speech_valid_clsattnbndkmeans/data_dict_bnd.pkl')
#parser.add_argument('--tsv_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/val2.tsv')
#parser.add_argument('--boundary_json', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/val2_vad_syllable.json')
# Final result (20ms): 22.198080753213834 43.48674292808371 29.392549971530464 95.90316573556797 -9.545148949001758 
# Final result (50ms): 38.461886655803006 75.34805356034407 50.927507566903415 95.90316573556797 7.866888849255538
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/syllablelm_hubert_base_ls960_ft_bnd/val2_data_bnd_dict.pkl')

# Final result (20ms): 21.388415363969386 52.29227631462269 30.359349258649093 144.48878247760933 -44.032333603293154
# Final result (20ms): 35.83475390809184 87.61195353374124 50.8649093904448 144.48878247760928 -27.973768573670597
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/syllablelm_hubert_large_ll60k_ft_bnd/val2_data_bnd_dict.pkl')

# Final result (20ms): 21.388415363969386 52.29227631462269 30.359349258649093 144.48878247760933 -44.032333603293154
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/syllablelm_hubert_large_ll60k_ft_bnd_delta-0.0038/val2_data_bnd_dict.pkl')

# Final result (20ms): 28.957023060796644 29.600857219394587 29.275400715326533 2.2234128047147195 39.10692673917564
# Final result (50ms): 55.95737246680643 57.20153585141531 56.572614474323316 2.2234128047146973 62.654273138548014
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/sylboost_data2vec2_base_ft_bnd/val2_data_bnd_dict.pkl')

# Final result (20ms): 28.108407956256436 31.672470756317527 29.78419682592997 12.679703544959375 36.61258185621665
# Final result (50ms): 53.546239797131314 60.33574426288062 56.73860105802335 12.679703544959375 60.67278441228837
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/sylboost_hubert_base_ls960_ft_bnd/val2_data_bnd_dict.pkl')

# Final result (20ms): 28.071539657853812 29.011518885614784 28.53379001449084 3.3485132601124956 38.18419779077643
# Final result (50ms): 54.622429583549334 56.4514688811501 55.521889957405705 3.3485132601124956 61.58085217398911
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/sylboost_hubert_large_ll60k_ft_bnd/val2_data_bnd_dict.pkl')

# Final result (20ms): 25.37449524553862 17.394410215197787 20.63996609451155 -31.449236538976688 37.718651037270156
# Final result (50ms): 52.35117884590335 35.88713277971247 42.58317440135622 -31.4492365389767 52.74622108414769
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/sylboost_hubert_large_ll60k_ft_bnd_delta-0.2/val2_data_bnd_dict.pkl')

# Final result (20ms): 26.781294776605446 18.358782034110188 21.784276329730876 -31.4492365389767 38.509903840006885
# Final result (50ms): 54.78702618210238 37.556924725421915 44.5645263827082 -31.449236538976688 54.08421963293908
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/sylboost_data2vec2_base_ft_bnd_delta-0.1/val2_data_bnd_dict.pkl')

# mergethres=0.4
# Final result (20ms): 30.320781032078102 29.11867130993839 29.70757037441924 -3.964639699973216 40.84531764025303
# Final result (50ms): 57.86145978614597 55.567461380480395 56.69126355106131 -3.9646396999732048 63.387903640733825
# parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/sylboost_bnd/val2_data_bnd_dict.pkl')

# Final result (20ms): 42.96640767987238 54.65502598770256 48.11095350921173 27.20408556125511 47.91034348234622
# Final result (50ms): 62.66103589970018 79.7073977194234 70.16370108180467 27.20408556125511 66.23791253912724
parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/data/LibriSpeech_vad/sylboost_bnd/valid_true_data_bnd_dict.pkl')

#parser.add_argument('--tsv_path', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/tsvs/valid.tsv')
#parser.add_argument('--boundary_json', type=str, default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/valid_vad_syllable.json')
#parser.add_argument('--seg_path', type=str, default='/data/sls/scratch/limingw/data/LibriSpeech_vad/syllablelm_bnd/valid_data_bnd_dict.pkl')
args = parser.parse_args()

frames_per_embedding = 160 #(not 320), instead of multiplying by 2 later
print("frame duration (s): %f" % (frames_per_embedding/16000), flush=True)

wavs, bounds, fns = data_loader.get_data_buckeye(args.tsv_path, args.boundary_json, load_wav=False)
with open(args.seg_path, 'rb') as f:
    pred_bound_dict = pickle.load(f)

ref_bounds = []
seg_bounds = []
for idx in tqdm(list(range(len(wavs)))):
    pred_bound = pred_bound_dict[fns[idx]]['seg_bound']

    if len(bounds[idx]) > 1:
        audio_len = pred_bound.shape[-1]
        pred_boundaries = np.nonzero(pred_bound)[0][1:]

        boundaries = np.array(data_loader.get_bounds(bounds[idx])) // frames_per_embedding
        boundaries = np.minimum(boundaries[1:], audio_len-1)

        print('idx, fn, boundaries, pred_boundaries:', idx, fns[idx], boundaries, pred_boundaries)  # XXX

        ref_bound = np.zeros(audio_len)
        seg_bound = np.zeros(audio_len)
        ref_bound[boundaries] = 1
        seg_bound[pred_boundaries] = 1
        ref_bound[-1] = 1
        seg_bound[-1] = 1
        
        ref_bounds.append(ref_bound)
        seg_bounds.append(seg_bound)

precision, recall, f = eval_segmentation.score_boundaries(ref_bounds, seg_bounds, 2)
os = eval_segmentation.get_os(precision, recall)*100
r_val = eval_segmentation.get_rvalue(precision, recall)*100
print("Final result (20ms):", precision*100, recall*100, f*100, os, r_val)

precision, recall, f = eval_segmentation.score_boundaries(ref_bounds, seg_bounds, 5)
os = eval_segmentation.get_os(precision, recall)*100
r_val = eval_segmentation.get_rvalue(precision, recall)*100
print("Final result (50ms):", precision*100, recall*100, f*100, os, r_val)
