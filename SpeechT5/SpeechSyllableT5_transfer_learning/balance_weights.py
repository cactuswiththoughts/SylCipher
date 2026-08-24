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


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', default='/data/sls/scratch/limingw/junruin2_satori/data/hubert/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip_v2/checkpoint_best.pt')
parser.add_argument('--src_dir', default='/data/sls/scratch/limingw/junruin2_satori/data/hubert/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048')
parser.add_argument('--out_ckpt_path', default='/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mdcc_vad/feat_xeus_L18_sylber_bnd/top_608/km_dir_608/sylcipher_transfer_aishell_3/checkpoint_best.pt')
args = parser.parse_args()

ckpt_path = args.ckpt_path
src_dir = args.src_dir
out_ckpt_path = args.out_ckpt_path
Path(out_ckpt_path).parent.mkdir(exist_ok=True, parents=True) 

src_dir = Path(src_dir)
# Load the pretrained model weights
ckpt = torch.load(ckpt_path)
new_ckpt = deepcopy(ckpt)

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
logging.info(f"text dim: {ckpt['model'][text_keys[0]]}")

src_params_list = [ckpt['model'][k] for k in speech_keys]
for k, p in zip(speech_keys, src_params_list):
    p_new = torch.cat([p, p[-1:], p[-1:]])
    logging.info(f'p.shape: {p.shape}, p_new.shape: {p_new.shape}')
    new_ckpt['model'][k] = p_new

torch.save(new_ckpt, out_ckpt_path)
