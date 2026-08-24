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
import json
from fairseq.tasks.hubert_pretraining import LabelEncoder
from fairseq.data import encoders
import random
NONWORD=('<OOV>', '<unk>')


def load_raw_label(label_path, file_list):
    
    with open(label_path, 'r') as f:
        label_json = json.load(f)
        
    print(len(label_json))
    wav2trans = {}
    for key, value in label_json.items():
        wav_fn = key
        wav_trans = " ".join([v[2] for v in value])
        wav2trans[wav_fn] = wav_trans
    
    labels = []
        
    for file in file_list:
        labels.append(wav2trans[file])
        
    return labels

def load_label(label_path, file_list, text_dict):
    
    with open(label_path, 'r') as f:
        label_json = json.load(f)
        
    print(len(label_json))
    wav2trans = {}
    for key, value in label_json.items():
        wav_fn = key
        wav_trans = " ".join([v[2] for v in value if v[2] in text_dict])
        wav2trans[wav_fn] = wav_trans
    
    labels = []
        
    for file in file_list:
        labels.append(wav2trans[file])
        
    return labels

def load_unpaired_text(text_path, max_lines = None, seed = 0):
    
    random.seed(0)
    with open(text_path, 'r') as f:
        text_lines = f.readlines()
        
    labels = []
    for l in text_lines:
        labels.append(l.strip())
    
    print(max_lines)
    if max_lines is not None:
        random.shuffle(labels)
        return labels[:max_lines]
        
    return labels

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
    parser.add_argument(
        "--postfix",
        default="",
        type=str,
        help="postfix",
    )
    parser.add_argument(
        "--tgt-enc-layer",
        default=0,
        type=int,
        help="encoder embedding layer",
    )
    parser.add_argument(
        "--use-discrete-labels",
        default=0,
        type=int,
        help="use discrete labels (set it to 1) or not",
    )
    parser.add_argument(
        "--use-unpaired-text",
        default=0,
        type=int,
        help="use unpaired (set it to 1) or not",
    )
    parser.add_argument(
        "--text-path",
        default="",
        type=str,
        help="Reference text path",
    )
    parser.add_argument(
        "--speech-prenet-mode",
        default='policy_loss',
        choices={'policy_pretrain', 'policy_loss'},
    )

def calc_uer(hf, rf):

    errs = 0
    count = 0
    # h and r are lists
    for h, r in zip(hf, rf):
        errs += editdistance.eval(r.strip().split(), h.strip().split())
        count += len(r.strip().split())
        
    print(errs, count)
    return errs / count


from itertools import groupby
def calc_uer_remove_rep(hf, rf):

    errs = 0
    count = 0
    # h and r are lists
    for h, r in zip(hf, rf):
        errs += editdistance.eval(r.strip().split(), [key for key, _group in groupby(h.strip().split())])
        count += len(r.strip().split())

    return errs / count

def calc_uer_remove_rep_and_oov(hf, rf):

    errs = 0
    count = 0
    # h and r are lists
    for h, r in zip(hf, rf):
        errs += editdistance.eval([rr for rr in r.strip().split() if rr != '<OOV>'], [key for key, _group in groupby(h.strip().split()) if key != '<OOV>'])
        count += len(r.strip().split())

    return errs / count

def split_token(w):
    if w in NONWORD:
        return [w]
    else:
        return list(w)

if __name__ == "__main__":
    import argparse
    import editdistance
    # split, ckpt_path, data_path, layer, pooling_type, feat_dir
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    
    module_options = edict({'user_dir': args.user_dir})
    utils.import_user_module(module_options)
    
    (model, cfg, task) = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.ckpt_path])
    model = model[0].eval().to("cuda:0")
    dicts = OrderedDict()
    dicts["text"] = Dictionary.load(op.join(args.data, "dict.txt"))
    mask_idx = dicts["text"].add_symbol("<mask>")
    blank_symbol_idx = dicts["text"].add_symbol("<ctc_blank>")
    
    speech_split, text_split = args.split.split('|')

    text_dict_symbols = set(dicts["text"].symbols)

    filenames = []
    if os.path.exists(os.path.join(args.data, f'{speech_split}.files')): 
        with open(os.path.join(args.data, f'{speech_split}.files'), 'r') as f:
            file_lines = f.readlines()
    else:
        with open(os.path.join(args.data, f'{speech_split}.tsv'), 'r') as f:
            file_lines = f.readlines()
            root = file_lines.pop(0)

    for idx, file_line in enumerate(file_lines):
        cur_filename = file_line.strip().split('\t')[0]
        filenames.append(cur_filename)

    cur_offset = 0
    all_feats_len = []
    offset = []
    with open(os.path.join(args.data, f'{speech_split}.lengths'), 'r') as len_f:
        for idx, line in enumerate(len_f):
            length = int(line.strip())
            all_feats_len.append(length)
            offset.append(cur_offset)
            cur_offset += length

    all_feats = np.load(os.path.join(args.data, f'{speech_split}.npy'), mmap_mode='r')
    print(len(filenames), all_feats.shape)
    
    if args.text_path:
        print('Evaluate using full-coverage (no <OOV>) text')
        label_list = load_raw_label(args.text_path, filenames)
    else:
        label_list = load_label(os.path.join(args.data, speech_split, 'meta_data.json'), filenames, text_dict_symbols)
    label_processor = LabelEncoder(dicts["text"])
    tokenizer = encoders.build_tokenizer(Namespace(**{"tokenizer": "space"}))
    
    max_text_positions = args.max_text_positions
    
    gt_text_strings = []
    gt_char_strings = []
    all_outputs = []
    all_char_outputs = []
    print(f'Speech dataset size: {len(all_feats_len)}; Text dataset size: {len(label_list)}')
    with torch.no_grad():
        for idx in range(len(all_feats_len)):
            cur_fn = filenames[idx]
            if idx % 1000 == 0:
                print(f'Processed {idx} samples')
             
            speech_start = offset[idx]
            speech_end = offset[idx] + all_feats_len[idx]
            speech_sample = all_feats[speech_start:speech_end]

            speech_sample = torch.from_numpy(speech_sample).to("cuda:0").unsqueeze(-1).unsqueeze(0)

            speech_padding_mask = torch.BoolTensor(speech_sample.shape[:2]).fill_(False).to("cuda:0")
            speech_encoder_padding_mask = speech_padding_mask

            output = model(idx, speech_sample, speech_padding_mask, dense_x_only=True)
            
            logits = output['logits']
            padding_mask = output['padding_mask']
            output_tokens = logits.argmax(-1).cpu().squeeze(0)
            speech_to_text_output = dicts['text'].string(output_tokens)
            
            all_outputs.append(speech_to_text_output)
            speech_to_char_output = ' '.join([c for w in speech_to_text_output.split() for c in split_token(w)])
            all_char_outputs.append(speech_to_char_output)
            text_sample = label_list[idx]
            
            text_string = text_sample
            gt_text_strings.append(text_string)
            char_string = ' '.join([c for w in text_string.split() for c in split_token(w)])
            gt_char_strings.append(char_string)
            if idx < 10:
                print('GT:', text_string)
                print('Pred:', speech_to_text_output)
                print('GT (char):', char_string)
                print('Pred (char):', speech_to_char_output)
            
    print(len(all_outputs), len(filenames), len(gt_text_strings))
    uer_1 = calc_uer_remove_rep_and_oov(all_outputs, gt_text_strings)
    uer_2 = calc_uer_remove_rep(all_outputs, gt_text_strings)
    uer_3 = calc_uer(all_outputs, gt_text_strings)

    cer_1 = calc_uer_remove_rep_and_oov(all_char_outputs, gt_char_strings)
    cer_2 = calc_uer_remove_rep(all_char_outputs, gt_char_strings)
    cer_3 = calc_uer(all_char_outputs, gt_char_strings)

    print('WER:', uer_1, uer_2, uer_3)
    print('CER:', cer_1, cer_2, cer_3)
    os.makedirs(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn'), exist_ok=True)
    with open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn', f'{args.postfix}_preds.txt'), 'w') as f,\
        open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn', f'{args.postfix}_preds_char.txt'), 'w') as f_char:
        for cur_output_pred, cur_char_output_pred, cur_output_fn in zip(all_outputs, all_char_outputs, filenames):
            f.write(cur_output_fn + '\t' + cur_output_pred + '\n')
            f_char.write(cur_output_fn + '\t' + cur_char_output_pred + '\n')

    with open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn', f'{args.postfix}_ref.txt'), 'w') as f,\
        open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn', f'{args.postfix}_ref_char.txt'), 'w') as f_char:
        for cur_output_gt, cur_char_output_gt, cur_output_fn in zip(gt_text_strings, gt_char_strings, filenames):
            f.write(cur_output_fn + '\t' + cur_output_gt + '\n')
            f_char.write(cur_output_fn + '\t' + cur_char_output_gt + '\n')
