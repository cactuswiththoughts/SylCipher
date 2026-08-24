import logging
import os.path as op
from argparse import Namespace
from collections import OrderedDict
import pickle

from matplotlib.pyplot import text

import torch
from fairseq.data import (
    Dictionary, 
    encoders, 
    PrependTokenDataset,
    AppendTokenDataset, 
    data_utils, 
    StripTokenDataset,
    TokenBlockDataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq import utils
import fairseq
from easydict import EasyDict as edict
import os
import pickle
import numpy as np


def add_args(parser):
    parser.add_argument("data", help="manifest root path")
    
    parser.add_argument(
        "--max-speech-sample-size",
        default=None,
        type=int,
        metavar="N",
        help="max speech sample size",
    )
    parser.add_argument(
        "--min-speech-sample-size",
        default=None,
        type=int,
        metavar="N",
        help="min speech sample size",
    )
    parser.add_argument(
        "--max-speech-positions",
        default=4000,
        type=int,
        metavar="N",
        help="max number of tokens in the source sequence",
    )
    parser.add_argument(
        "--max-text-positions",
        default=450,
        type=int,
        metavar="N",
        help="max number of tokens in the target sequence",
    )
    parser.add_argument(
        "--split",
        default="speech_train|text_train",
        type=str,
        help="split to use",
    )
    parser.add_argument(
        "--ckpt-path",
        default="",
        type=str,
        help="checkpoint path",
    )
    parser.add_argument(
        "--user-dir",
        default="",
        type=str,
        help="user dir path",
    )

if __name__ == "__main__":
    import argparse
    # split, ckpt_path, boundary_path, data_path, layer, pooling_type, feat_dir
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    speech_split, text_split = args.split.split('|')
    which_set_flag = text_split.split('_', maxsplit =1)[1]
    
    filenames = []
    with open(os.path.join(args.data, speech_split, 'file_list.txt'), 'r') as f:
        file_lines = f.readlines()
    for idx, file_line in enumerate(file_lines):
        filenames.append(file_line.strip())
    
    all_feats = np.load(os.path.join(args.data, speech_split, 'data.npy'), mmap_mode='r')
    all_feats_len = []
    with open(os.path.join(args.data, speech_split, 'data.lengths'), 'r') as f:
        length_lines = f.readlines()
    for len_line in length_lines:
        all_feats_len.append(int(len_line.strip()))
        
    offset = [0] + list(np.cumsum(all_feats_len)[:-1])
    
    module_options = edict({'user_dir': args.user_dir})
    utils.import_user_module(module_options)
    
    (model, cfg, task) = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.ckpt_path])
    model = model[0].eval().to("cuda:0")
    policy_network = model.speech_encoder_prenet.policy_network
    policy_network_logits = model.speech_encoder_prenet.policy_network_logits
    dicts = OrderedDict()
    dicts["audio"] = Dictionary.load(op.join(args.data, "dict.audio.txt"))
    speeech_mask_idx = dicts["audio"].add_symbol("<mask>")
    
    
    all_outputs = {}
    
    with torch.no_grad():
        for idx in range(len(offset)):
            if idx % 1000 == 0:
                print(f'Processed {idx} samples')
            
            start = offset[idx]
            end = start + all_feats_len[idx]
            speech_sample = torch.from_numpy(all_feats[start:end].copy()).to("cuda:0").unsqueeze(0)
            speech_filename = filenames[idx]

            speech_sample_length = speech_sample.numel()
            padding_mask = torch.BoolTensor(speech_sample.shape[:-1]).fill_(False).to("cuda:0")
            boundary_logits = policy_network_logits(policy_network(speech_sample)).squeeze()
            boundary_preds = torch.zeros(boundary_logits.size())
            boundary_preds[boundary_logits > 0] = 1
            
            all_outputs[speech_filename] = boundary_preds
            
    with open(op.join(os.path.dirname(args.ckpt_path), f'predicted_boundaries_{which_set_flag}.pkl'), 'wb') as f:
        pickle.dump(all_outputs, f)
        
        

# python dump_boundary.py "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/" --ckpt-path "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/model_codebook_0.3_from_gradsegclsattnbndkmeans_diff_bnd_retry/checkpoint_best.pt" --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5" --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true"

# python dump_boundary.py "/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/" --ckpt-path "/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattnbndkmeans_diff_bnd_1/checkpoint_best_62.126.pt" --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5" --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true"

# python dump_boundary.py "/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/" --ckpt-path "/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattnbndkmeans_diff_bnd_1/checkpoint_best_62.126.pt" --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd/speecht5" --split "discrete_speech_train_clsattnbndkmeans|text_train"


# python dump_boundary.py "/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/" --ckpt-path "/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_orikmeans_encoder_only_retry_2024_2l_diff_bnd_retry/checkpoint_best.pt" --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_train_clsattnbndkmeans|text_train"

# python dump_boundary.py "/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/" --ckpt-path "/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/model_codebook_0.3_clsattn_kmeans_encoder_only_retry_2024_2l_large_batch_diff_bnd_no_codebook_prob/checkpoint_best.pt" --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5" --split "discrete_speech_train_clsattnbndkmeans|text_train"