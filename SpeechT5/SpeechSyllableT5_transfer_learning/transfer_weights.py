import argparse
import editdistance
import logging
import numpy as np
import os
import sys
import torch
from copy import deepcopy
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("transfer_prenet_postnet_weights")


class ApplyKmeans(object):
    def __init__(self, C_np):
        self.C_np = C_np.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def init_audio_emb(C_tgt, C_src, W_src_list, n_special=4):
    km_src = ApplyKmeans(C_src)

    labels = km_src(C_tgt)
    print(f'Use {len(set(labels))} of {C_src.shape[0]} centroids')

    W_new_list = []
    for W in W_src_list:
        new_size = (len(labels)+n_special+1,)+W.shape[1:]
        W_new = W.new_zeros(*new_size)
        W_new[n_special:-1] = W[labels+n_special]
        print(f'W.shape: {W.shape}, W_new.shape: {W_new.shape}')
        W_new_list.append(W_new)

    return W_new_list


def init_text_emb(tgt_tokens, src_tokens, W_src_list, n_special=4):
    labels = []
    overlap = 0
    for token in tgt_tokens:
        if token in src_tokens:
            labels.append(src_tokens.index(token))
            overlap += 1
        else:
            # Match by edit distance
            dist = np.asarray([editdistance.eval(src_token, token) for src_token in src_tokens])
            print(f'{token} -> {src_tokens[np.argmin(dist)]}')
            logger.info(f'{token} -> {src_tokens[np.argmin(dist)]}')
            labels.append(np.argmin(dist)+n_special)
    print(f'{overlap} out of {len(src_tokens)}+{len(tgt_tokens)} tokens overlap')
    logger.info(f'{overlap} out of {len(src_tokens)}+{len(tgt_tokens)} tokens overlap')

    W_new_list = []
    for W in W_src_list:
        new_size = (len(labels)+n_special+2,)+W.shape[1:]
        W_new = W.new_zeros(*new_size)
        W_new[n_special:-2] = W[labels]
        print(f'W.shape: {W.shape}, W_new.shape: {W_new.shape}')
        W_new_list.append(W_new)

    return W_new_list


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', default='/data/sls/scratch/limingw/junruin2_satori/data/hubert/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip_v2/checkpoint_best.pt')
parser.add_argument('--src_dir', default='/data/sls/scratch/limingw/junruin2_satori/data/hubert/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048')
parser.add_argument('--tgt_dir', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mdcc_vad/feat_xeus_L18_sylber_bnd/top_608/km_dir_608')
parser.add_argument('--out_ckpt_path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mdcc_vad/feat_xeus_L18_sylber_bnd/top_608/km_dir_608/sylcipher_transfer_aishell_3/checkpoint_best.pt')
args = parser.parse_args()

ckpt_path = args.ckpt_path
src_dir = args.src_dir
tgt_dir = args.tgt_dir
out_ckpt_path = args.out_ckpt_path
Path(out_ckpt_path).parent.mkdir(exist_ok=True, parents=True) 

src_dir = Path(src_dir)
tgt_dir = Path(tgt_dir)
# Load the pretrained model weights
ckpt = torch.load(ckpt_path)
new_ckpt = deepcopy(ckpt)

# Load cluster centroids and token dictionaries for the pretrained and new models
C_src = np.load(str(src_dir / 'seg_feats' / 'km_model.npy'))
C_tgt = np.load(str(tgt_dir / 'seg_feats' / 'km_model.npy'))

with open(src_dir / 'dict.txt', 'r') as f_src,\
    open(tgt_dir / 'dict.txt', 'r') as f_tgt:
    src_tokens = [l.strip().split()[0] for l in f_src.read().strip().split('\n')]
    tgt_tokens = [l.strip().split()[0] for l in f_tgt.read().strip().split('\n')]

# Transfer weights for the speech prenet and postnet
speech_keys = [
    'encoder.src_proj.weight',
    'encoder.src_proj.bias', 
    'speech_encoder_prenet.encoder_prenet.0.weight',
    'speech_predictor.predictor_net.weight',
    'speech_predictor.predictor_net.bias',
    'speech_encoder_postnet.label_embs_concat',
    'speech_decoder_prenet.embed_tokens.weight',
    'speech_decoder_postnet.output_projection.weight',
]
src_params_list = [ckpt['model'][k] for k in speech_keys]
tgt_params_list = init_audio_emb(C_tgt, C_src, src_params_list)

new_ckpt['model']['speech_encoder_prenet.quantizer.embedding'] = torch.from_numpy(C_tgt)
for k, p in zip(speech_keys, tgt_params_list):
    new_ckpt['model'][k] = p 

# Transfer weights for the text prenet and postnet
text_keys = [
    'encoder.proj.weight',
    'encoder.proj.bias', 
    'text_encoder_prenet.encoder_prenet.0.weight',
    'text_predictor.predictor_net.weight',
    'text_predictor.predictor_net.bias',
    'text_encoder_postnet.label_embs_concat',
    'text_decoder_prenet.embed_tokens.weight',
    'text_decoder_postnet.output_projection.weight',
]
src_params_list = [ckpt['model'][k] for k in text_keys]
tgt_params_list = init_text_emb(tgt_tokens, src_tokens, src_params_list)
for k, p in zip(text_keys, tgt_params_list):
    new_ckpt['model'][k] = p 

torch.save(new_ckpt, out_ckpt_path)
