
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
import math

def load_label(label_path, file_list, text_dict):
    
    with open(label_path, 'r') as f:
        label_json = json.load(f)
        
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
        "--unpaired-text-path",
        default="",
        type=str,
        help="unpaired text path",
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


def word_starts_helper(source, mask_whole_word=None):
    if mask_whole_word is not None:
        is_word_start = mask_whole_word.gather(0, source)
    else:
        is_word_start = torch.ones(source.size())
    is_word_start[0] = 0
    is_word_start[-1] = 0
    return is_word_start

def add_whole_word_mask(source, vocab, mask_idx, mask_whole_word, p = 0.3, mask_span_distributiion= None, random_ratio = 0.1, replace_length = -1, iid_noise_target=False, uni_mask_idxs=None):
    source_ori = source.clone()
    is_word_start = word_starts_helper(source, mask_whole_word)
    num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
    num_inserts = 0
    if num_to_mask == 0:
        return source

    if mask_span_distribution is not None:
        lengths = mask_span_distribution.sample(sample_shape=(num_to_mask,))

        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat(
                [
                    lengths,
                    mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                ],
                dim=0,
            )
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return source, source_ori, None
        # if num_to_mask == 0:
        #     return self.add_insertion_noise(source, num_inserts / source.size(0))

        assert (lengths > 0).all()
    else:
        lengths = torch.ones((num_to_mask,)).long()
    assert is_word_start[-1] == 0
    word_starts = is_word_start.nonzero(as_tuple=False)
    indices = word_starts[
        torch.randperm(word_starts.size(0))[:num_to_mask]
    ].squeeze(1)
    mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_ratio

    source_length = source.size(0)
    assert source_length - 1 not in indices
    to_keep = torch.ones(source_length, dtype=torch.bool)
    is_word_start[
        -1
    ] = 255  # acts as a long length, so spans don't go over the end of doc
    if replace_length == 0:
        to_keep[indices] = 0
    else:
        # keep index, but replace it with [MASK]
        source[indices] = mask_idx
        source[indices[mask_random]] = torch.randint(
            1, len(vocab), size=(mask_random.sum(),)
        ).to(source.device)

    if mask_span_distribution is not None:
        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = mask_idx
                source[indices[mask_random]] = torch.randint(
                    1, len(vocab), size=(mask_random.sum(),)
                ).to(source.device)
    else:
        # A bit faster when all lengths are 1
        while indices.size(0) > 0:
            uncompleted = is_word_start[indices + 1] == 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = mask_idx
                source[indices[mask_random]] = torch.randint(
                    1, len(vocab), size=(mask_random.sum(),)
                ).to(source.device)

            assert source_length - 1 not in indices

    if not iid_noise_target:
        unmasked_source = source_ori[to_keep]
        source = source[to_keep]
        target = None
        assert len(unmasked_source) == len(source)
    else:
        print('Warning! iid_noise_target is not False')
        ## Prepare source
        source_mask_idx = (source == mask_idx).nonzero().view(-1)
        source[source_mask_idx] = uni_mask_idxs[:source_mask_idx.size(0)]
        source = source[to_keep]

        ## Prepare target
        to_keep[source_mask_idx] = 0

        # source_mask_idx: from [a, b, c, ...] to [a, b + 1, c + 2, ...]
        source_mask_idx = source_mask_idx + torch.arange(source_mask_idx.size(0))
        # target: source_length + mask_length
        target = source_ori.new_zeros(source_mask_idx.size(0) + source_ori.size(0))
        # target: [0, 0, 0, X, 0, 0, Y, ....]
        target[source_mask_idx] = uni_mask_idxs[:source_mask_idx.size(0)]

        target_to_keep = to_keep.new_zeros(source_mask_idx.size(0) + source_ori.size(0))

        # Copy original value to target and target_to_keep
        target_to_keep[target == 0] = to_keep
        target_to_keep[-1] = 0
        target[target == 0] = source_ori

        target = target[~target_to_keep]

    # if num_inserts > 0:
    #     source, unmasked_source = self.add_insertion_noise(source, unmasked_source, num_inserts / source.size(0))

    return source, unmasked_source, target


# def get_whole_word_mask(dictionary):
#     bpe = encoders.build_tokenizer(Namespace(**{"tokenizer": "space"}))
#     if bpe is not None:

#         def is_beginning_of_word(i):
#             if i < dictionary.nspecial:
#                 # special elements are always considered beginnings
#                 return True
#             tok = dictionary[i]
#             if tok.startswith("madeupword"):
#                 return True
#             try:
#                 return bpe.is_beginning_of_word(tok)
#             except ValueError:
#                 return True

#         mask_whole_words = torch.ByteTensor(
#             list(map(is_beginning_of_word, range(len(dictionary))))
#         )
#         return mask_whole_words
#     return None
    
if __name__ == "__main__":
    import argparse
    import editdistance
    # split, ckpt_path, boundary_path, data_path, layer, pooling_type, feat_dir
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    
    random_ratio = 0.1
    mask_ratio = 0.3
    mask_span_distribution = None
    mask_length = "span-poisson"
    poisson_lambda = 3.5
    if mask_length == "span-poisson":
        _lambda = poisson_lambda

        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        mask_span_distribution = torch.distributions.Categorical(ps)
        
    iid_noise_target = False
    uni_mask_idxs = None
    replace_length = -1

    
    module_options = edict({'user_dir': args.user_dir})
    utils.import_user_module(module_options)
    
    (model, cfg, task) = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.ckpt_path])
    model = model[0].eval().to("cuda:0")
    dicts = OrderedDict()
    dicts["text"] = Dictionary.load(op.join(args.data, "dict.txt"))
    mask_idx = dicts["text"].add_symbol("<mask>")
    blank_symbol_idx = dicts["text"].add_symbol("<ctc_blank>")
    dicts["audio"] = Dictionary.load(op.join(args.data, "dict.audio0.txt"))
    speech_mask_idx = dicts["audio"].add_symbol("<mask>")
    # print('cfg:')
    # print(cfg)
    speech_mask_whole_word = get_whole_word_mask(cfg.task, dicts["audio"])
    # print(speech_mask_whole_word)
    speech_padding_idx = dicts["audio"].pad()
    
    
    text_mask_whole_word = get_whole_word_mask(cfg.task, dicts["text"])
    # print(text_mask_whole_word)
    text_mask_idx = mask_idx
    text_padding_idx = dicts["text"].pad()

    speech_split, text_split = args.split.split('|')

    bad_idxs = []
    discrete_labels = []
    text_dict_symbols = set(dicts["text"].symbols)
    

    bad_files = []
    if 'clsattnbndkmeans' in speech_split:
        with open(os.path.join('/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/', speech_split, 'bnd_errors.txt'), 'r') as f:
            bad_lines = f.readlines()
    elif '_gt_kmeans' in speech_split:
        with open(os.path.join('/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/', speech_split.replace('_gt_kmeans', '_clsattnbndkmeans'), 'bnd_errors.txt'), 'r') as f:
            bad_lines = f.readlines()
    else:
        with open(os.path.join('/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_no_sil_wav2bnd_gradseg/top_1024/km_dir_1024/', speech_split + '_clsattnbndkmeans', 'bnd_errors.txt'), 'r') as f:
            bad_lines = f.readlines()

    for bl in bad_lines:
        bad_file = bl.strip().split(maxsplit = 1)[0]
        bad_files.append(bad_file)

    bad_files = set(bad_files)
       
    filenames = []
    with open(os.path.join(args.data, speech_split, 'file_list.txt'), 'r') as f:
        file_lines = f.readlines()
    for idx, file_line in enumerate(file_lines):
        cur_filename = file_line.strip()
        if cur_filename not in bad_files:
            filenames.append(cur_filename)
        else:
            bad_idxs.append(idx)
            
    bad_idxs = set(bad_idxs)     
    

    with open(os.path.join(args.data, speech_split, 'data.km')) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx not in bad_idxs:
                discrete_labels.append(line.strip())


    cur_offset = 0
    all_feats_len = []
    offset = []
    with open(os.path.join(args.data, speech_split, 'data.lengths'), 'r') as len_f:
        for idx, line in enumerate(len_f):
            length = int(line.strip())
            if idx not in bad_idxs:
                all_feats_len.append(length)
                offset.append(cur_offset)
            cur_offset += length

    all_feats = np.load(os.path.join(args.data, speech_split, 'data.npy'), mmap_mode='r')


              
    # filenames = []
    # with open(os.path.join(args.data, speech_split, 'file_list.txt'), 'r') as f:
    #     file_lines = f.readlines()
    # for idx, file_line in enumerate(file_lines):
    #     if idx not in bad_idxs:
    #         filenames.append(file_line.strip())


    with open(os.path.join(args.data, speech_split, 'data_dict_bnd.pkl'), 'rb') as fb:
        boundary_dict = pickle.load(fb)
    
    self_boundaries = []
    # for idx, filename in enumerate(self.filenames):
    #     if self.all_feats_len[idx] == 0:
    #         print('bad file encountered')
    #     boundary_arr = np.zeros(self.all_feats_len[idx]).astype('int32')
    #     boundaries = boundary_dict[filename]["boundaries"]
    #     for boundary_seg in boundaries:
    #         boundary_arr[int(50*boundary_seg[0]): int(50 * boundary_seg[1])] = 1
    #     self.boundaries.append(boundary_arr)
    
    gt_text_strings = []

    for idx, filename in enumerate(filenames):
        boundary_arr = np.zeros(all_feats_len[idx]).astype('int32')
        boundaries = boundary_dict[filename]
        
        boundary_locs = np.where(boundaries['seg_bound'] == 1)[0]
        # print(boundary_locs, len(curr_data))
        if len(boundary_locs) == 0:
            boundary_locs = np.arange(0, 2 * (all_feats_len[idx]), 26).astype('int32')
            # print('empty boundaries found!')
        if len(boundary_locs) == 1:
            boundary_locs = np.arange(0, 2 * (all_feats_len[idx]), 26).astype('int32')
            # print('length-one boundaries found!')
        if len(boundary_locs) == 1:
            boundary_locs = [0, 2 * (all_feats_len[idx]) - 1]
            boundary_locs = np.array(boundary_locs).astype('int32')
            # print('length-one boundaries found but due to short utt!')
        boundary_arr[boundary_locs // 2] = 1
        boundary_arr[0] = 1
        # boundary_arr[-1] = 1
        self_boundaries.append(boundary_arr)


    
    if args.use_unpaired_text == 1:
        label_list = load_unpaired_text(text_path = args.unpaired_text_path, max_lines = len(all_feats_len))
    else:
        label_list = load_label(os.path.join(args.data, speech_split, 'meta_data.json'), filenames, text_dict_symbols)
    label_processor = LabelEncoder(dicts["text"])
    tokenizer = encoders.build_tokenizer(Namespace(**{"tokenizer": "space"}))
    
    src_label_processor = LabelEncoder(dicts["audio"])
    src_tokenizer = encoders.build_tokenizer(Namespace(**{"tokenizer": "space"}))

        

    # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
    # text_bart_dataset = PrependTokenDataset(text_bart_dataset, dicts["text"].bos())
    
    
    # print(bart_dataset[0], len(bart_dataset[0]))
    # print(dicts["text"].string(bart_dataset[0]), len(dicts["text"].string(bart_dataset[0]).split()))
    
    max_text_positions = args.max_text_positions
    
    total_speech_correct = 0
    total_speech = 0
    total_text_correct = 0
    total_text = 0
    print(f'Speech dataset size: {len(all_feats_len)}; Text dataset size: {len(label_list)}')
    with torch.no_grad():
        for idx in range(len(all_feats_len)):
            cur_fn = filenames[idx]
            # if idx % 1000 == 0:
            #     print(f'Processed {idx} samples')
            if args.use_discrete_labels != 1:
                speech_start = offset[idx]
                speech_end = offset[idx] + all_feats_len[idx]
                speech_sample = all_feats[speech_start:speech_end]

                speech_sample = torch.from_numpy(speech_sample).to("cuda:0").unsqueeze(0)
                boundary_sample = torch.from_numpy(self_boundaries[idx]).to("cuda:0").unsqueeze(0)
                # print(speech_sample.size())
                speech_padding_mask = torch.BoolTensor(speech_sample.shape[:-1]).fill_(False).to("cuda:0")
                speech_encoder_padding_mask = speech_padding_mask
                speech_output, speech_meta_info = model(source = speech_sample, boundaries = boundary_sample, padding_mask = speech_padding_mask, task_name = 'speech_pretrain', speech_prenet_mode="policy_loss", feature_only=True,  tgt_enc_layer=args.tgt_enc_layer)
                speech_sample = speech_meta_info["src_tokens"].squeeze().detach()

            else:
                speech_sample = discrete_labels[idx]
                if src_tokenizer is not None:
                    speech_sample = src_tokenizer.encode(speech_sample)

                if src_label_processor is not None:
                    speech_sample = src_label_processor(speech_sample)
                    
            speech_sample = torch.cat([torch.LongTensor([dicts["audio"].bos()]),
                                        speech_sample,
                                        torch.LongTensor([dicts["audio"].eos()])])
            
            speech_sample = speech_sample.to("cuda:0")
            speech_sample_target = torch.clone(speech_sample)
            masked_speech_sample,_,_ = add_whole_word_mask(source =speech_sample, vocab=dicts["audio"], mask_idx=speech_mask_idx, mask_whole_word=speech_mask_whole_word, p = mask_ratio, mask_span_distributiion=mask_span_distribution, random_ratio = random_ratio, replace_length = replace_length, iid_noise_target=iid_noise_target, uni_mask_idxs=uni_mask_idxs)
            masked_speech_sample = masked_speech_sample.unsqueeze(0)
            speech_encoder_input = model.speech_encoder_prenet.encoder_prenet(masked_speech_sample)
            speech_encoder_padding_mask = torch.BoolTensor(masked_speech_sample.shape).fill_(False).to("cuda:0")

            encoder_output = model.encoder(speech_encoder_input, speech_encoder_padding_mask, tgt_layer=args.tgt_enc_layer)
            speech_output = encoder_output["encoder_out"][0].transpose(0, 1)
            
            speech_to_speech_output = model.speech_predictor(speech_output, speech_encoder_padding_mask).argmax(-1).squeeze(0)
            
            speech_preds = speech_to_speech_output[masked_speech_sample.squeeze() != speech_sample_target]
            speech_targets = speech_sample_target[masked_speech_sample.squeeze() != speech_sample_target]
            
            text_sample = label_list[idx]
            if tokenizer is not None:
                text_sample = tokenizer.encode(text_sample)

            if label_processor is not None:
                text_sample = label_processor(text_sample)
                
                
            text_sample = torch.cat([torch.LongTensor([dicts["text"].bos()]),
                                        text_sample,
                                        torch.LongTensor([dicts["text"].eos()])])
            
            
            text_sample = text_sample.to("cuda:0")
            text_sample_target = torch.clone(text_sample)
            masked_text_sample,_,_ = add_whole_word_mask(source = text_sample, vocab=dicts["text"], mask_idx=text_mask_idx, mask_whole_word=text_mask_whole_word, p = mask_ratio, mask_span_distributiion=mask_span_distribution, random_ratio = random_ratio, replace_length = replace_length, iid_noise_target=iid_noise_target, uni_mask_idxs=uni_mask_idxs)
            masked_text_sample = masked_text_sample.unsqueeze(0)
            text_encoder_input,_ = model.text_encoder_prenet(masked_text_sample)
            text_encoder_padding_mask = torch.BoolTensor(masked_text_sample.shape).fill_(False).to("cuda:0")

            text_encoder_output = model.encoder(text_encoder_input, text_encoder_padding_mask, tgt_layer=args.tgt_enc_layer)
            text_output = text_encoder_output["encoder_out"][0].transpose(0, 1)
            
            text_to_text_output = model.text_predictor(text_output, text_encoder_padding_mask).argmax(-1).squeeze(0)
            
            text_preds = text_to_text_output[masked_text_sample.squeeze() != text_sample_target]
            text_targets = text_sample_target[masked_text_sample.squeeze() != text_sample_target]
            
            
            # if idx < 10:
            #     print('GT Speech:', dicts["audio"].string(speech_sample_target))
            #     print('Pred Speech:', dicts["audio"].string(speech_to_speech_output))
            #     print('GT Text:', dicts["text"].string(text_sample_target))
            #     print('Pred Text:', dicts["text"].string(text_to_text_output))
            
            total_speech_correct += torch.sum(speech_preds == speech_targets).item()
            total_speech += len(speech_targets)
            
            total_text_correct += torch.sum(text_preds == text_targets).item()
            total_text += len(text_targets)
            
            
    print('Masked Speech Accu in %:', total_speech_correct / total_speech * 100)
    print('Masked Text Accu in %:', total_text_correct / total_text * 100)