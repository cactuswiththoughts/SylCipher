# import logging
# import os.path as op
# from argparse import Namespace
# from collections import OrderedDict
# import pickle

# from matplotlib.pyplot import text

# import torch
# from fairseq.data import (
#     Dictionary, 
#     encoders, 
#     PrependTokenDataset,
#     AppendTokenDataset, 
#     data_utils, 
#     StripTokenDataset,
#     TokenBlockDataset,
# )
# from fairseq.data.encoders.utils import get_whole_word_mask
# from fairseq import utils
# import fairseq
# from easydict import EasyDict as edict
# import os
# import pickle
# import numpy as np
# import json
# from fairseq.tasks.hubert_pretraining import LabelEncoder
# from fairseq.data import encoders

# def load_label(label_path, file_list, text_dict):
    
#     with open(label_path, 'r') as f:
#         label_json = json.load(f)
        
#     wav2trans = {}
#     for key, value in label_json.items():
#         wav_fn = key
#         wav_trans = " ".join([v[2] for v in value if v[2] in text_dict])
#         wav2trans[wav_fn] = wav_trans
    
#     labels = []
        
#     for file in file_list:
#         labels.append(wav2trans[file])
        
#     return labels

# def add_args(parser):
#     parser.add_argument("data", help="manifest root path")
    
#     parser.add_argument(
#         "--max-speech-sample-size",
#         default=None,
#         type=int,
#         metavar="N",
#         help="max speech sample size",
#     )
#     parser.add_argument(
#         "--min-speech-sample-size",
#         default=None,
#         type=int,
#         metavar="N",
#         help="min speech sample size",
#     )
#     parser.add_argument(
#         "--max-speech-positions",
#         default=4000,
#         type=int,
#         metavar="N",
#         help="max number of tokens in the source sequence",
#     )
#     parser.add_argument(
#         "--max-text-positions",
#         default=450,
#         type=int,
#         metavar="N",
#         help="max number of tokens in the target sequence",
#     )
#     parser.add_argument(
#         "--split",
#         default="speech_train|text_train",
#         type=str,
#         help="split to use",
#     )
#     parser.add_argument(
#         "--ckpt-path",
#         default="",
#         type=str,
#         help="checkpoint path",
#     )
#     parser.add_argument(
#         "--user-dir",
#         default="",
#         type=str,
#         help="user dir path",
#     )
#     parser.add_argument(
#         "--postfix",
#         default="",
#         type=str,
#         help="postfix",
#     )
#     parser.add_argument(
#         "--tgt-enc-layer",
#         default="",
#         type=int,
#         help="encoder embedding layer",
#     )


    
# if __name__ == "__main__":
#     import argparse
#     # split, ckpt_path, boundary_path, data_path, layer, pooling_type, feat_dir
#     parser = argparse.ArgumentParser()
#     add_args(parser)
#     args = parser.parse_args()
    
    
#     module_options = edict({'user_dir': args.user_dir})
#     utils.import_user_module(module_options)
    
#     (model, cfg, task) = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.ckpt_path])
#     model = model[0].eval().to("cuda:0")
#     dicts = OrderedDict()
#     dicts["text"] = Dictionary.load(op.join(args.data, "dict.txt"))
#     mask_idx = dicts["text"].add_symbol("<mask>")
#     blank_symbol_idx = dicts["text"].add_symbol("<ctc_blank>")
#     dicts["audio"] = Dictionary.load(op.join(args.data, "dict.audio0.txt"))
#     speeech_mask_idx = dicts["audio"].add_symbol("<mask>")

#     speech_split, text_split = args.split.split('|')

#     bad_idxs = []
#     text_dict_symbols = set(dicts["text"].symbols)

#     with open(os.path.join(args.data, speech_split, 'data.km')) as f:
#         lines = f.readlines()
#         for idx, line in enumerate(lines):
#             if len(line.strip().split()) < 3:
#                 bad_idxs.append(idx)


#     cur_offset = 0
#     all_feats_len = []
#     offset = []
#     with open(os.path.join(args.data, speech_split, 'data.lengths'), 'r') as len_f:
#         for idx, line in enumerate(len_f):
#             length = int(line.strip())
#             if idx not in bad_idxs:
#                 all_feats_len.append(length)
#                 offset.append(cur_offset)
#             cur_offset += length

#     all_feats = np.load(os.path.join(args.data, speech_split, 'data.npy'), mmap_mode='r')


        
#     bad_idxs = set(bad_idxs)        
#     filenames = []
#     with open(os.path.join(args.data, speech_split, 'file_list.txt'), 'r') as f:
#         file_lines = f.readlines()
#     for idx, file_line in enumerate(file_lines):
#         if idx not in bad_idxs:
#             filenames.append(file_line.strip())


#     with open(os.path.join(args.data, speech_split, 'data_dict_bnd.pkl'), 'rb') as fb:
#         boundary_dict = pickle.load(fb)
    
#     self_boundaries = []
#     # for idx, filename in enumerate(self.filenames):
#     #     if self.all_feats_len[idx] == 0:
#     #         print('bad file encountered')
#     #     boundary_arr = np.zeros(self.all_feats_len[idx]).astype('int32')
#     #     boundaries = boundary_dict[filename]["boundaries"]
#     #     for boundary_seg in boundaries:
#     #         boundary_arr[int(50*boundary_seg[0]): int(50 * boundary_seg[1])] = 1
#     #     self.boundaries.append(boundary_arr)


#     for idx, filename in enumerate(filenames):
#         boundary_arr = np.zeros(all_feats_len[idx]).astype('int32')
#         boundaries = boundary_dict[filename]
        
#         boundary_locs = np.where(boundaries['seg_bound'] == 1)[0]
#         # print(boundary_locs, len(curr_data))
#         if len(boundary_locs) == 0:
#             boundary_locs = np.arange(0, 2 * (all_feats_len[idx]), 26).astype('int32')
#             # print('empty boundaries found!')
#         if len(boundary_locs) == 1:
#             boundary_locs = np.arange(0, 2 * (all_feats_len[idx]), 26).astype('int32')
#             # print('length-one boundaries found!')
#         if len(boundary_locs) == 1:
#             boundary_locs = [0, 2 * (all_feats_len[idx]) - 1]
#             boundary_locs = np.array(boundary_locs).astype('int32')
#             # print('length-one boundaries found but due to short utt!')
#         boundary_arr[boundary_locs // 2] = 1
#         boundary_arr[0] = 1
#         # boundary_arr[-1] = 1
#         self_boundaries.append(boundary_arr)



#     label_list = load_label(os.path.join(args.data, speech_split, 'meta_data.json'), filenames, text_dict_symbols)
#     label_processor = LabelEncoder(dicts["text"])
#     tokenizer = encoders.build_tokenizer(Namespace(**{"tokenizer": "space"}))
        

#     # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
#     # text_bart_dataset = PrependTokenDataset(text_bart_dataset, dicts["text"].bos())
    
    
#     # print(bart_dataset[0], len(bart_dataset[0]))
#     # print(dicts["text"].string(bart_dataset[0]), len(dicts["text"].string(bart_dataset[0]).split()))
    
#     max_text_positions = args.max_text_positions
#     all_outputs = []
#     print(f'Speech dataset size: {len(all_feats_len)}; Text dataset size: {len(label_list)}')
#     with torch.no_grad():
#         for idx in range(len(all_feats_len)):
#             cur_fn = filenames[idx]
#             if idx % 1000 == 0:
#                 print(f'Processed {idx} samples')
#             speech_start = offset[idx]
#             speech_end = offset[idx] + all_feats_len[idx]
#             speech_sample = all_feats[speech_start:speech_end]
#             text_sample = label_list[idx]
            
#             text_string = text_sample

#             # source=None, aux_frame_clus=None, audio_tokens = None, discrete_batch = None, src_tokens=None, prev_output_tokens=None, random_label= None,random_src_tokens=None, boundaries = None, task_name=None, padding_mask=None, only_ctc=False, feature_only=False, tgt_enc_layer=None, input_type=None, output_type=None, speech_prenet_mode=None

#             speech_sample = torch.from_numpy(speech_sample).to("cuda:0").unsqueeze(0)
#             boundary_sample = torch.from_numpy(self_boundaries[idx]).to("cuda:0").unsqueeze(0)
#             # print(speech_sample.size())
#             speech_padding_mask = torch.BoolTensor(speech_sample.shape[:-1]).fill_(False).to("cuda:0")
#             speech_output, speech_meta_info = model(source = speech_sample, boundaries = boundary_sample, padding_mask = speech_padding_mask, task_name = 'speech_pretrain', speech_prenet_mode="policy_loss", feature_only=True,  tgt_enc_layer=args.tgt_enc_layer)
#             speech_string_list = (speech_meta_info["src_tokens"].cpu().squeeze() - 4).tolist()
#             speech_string_list = [str(s) for s in speech_string_list]
#             speech_string = " ".join(speech_string_list)

#             if tokenizer is not None:
#                 text_sample = tokenizer.encode(text_sample)

#             if label_processor is not None:
#                 text_sample = label_processor(text_sample)

#             text_sample = text_sample.to("cuda:0")
            
#             text_output = model(src_tokens = text_sample.unsqueeze(0), task_name = 'text_pretrain', feature_only=True, tgt_enc_layer=args.tgt_enc_layer)

#             # print(text_output.size(), text_string)
            
#             assert text_output.size(1) == len(text_string.split())
#             assert speech_output.size(1) == len(speech_string.split())
            
#             all_outputs.append((text_output.squeeze().cpu().numpy(), speech_output.squeeze().cpu().numpy(), text_string, speech_string, cur_fn))
            
#     os.makedirs(op.join(os.path.dirname(args.ckpt_path), f'all_feats_layer_{args.tgt_enc_layer}'), exist_ok=True)
#     with open(op.join(os.path.dirname(args.ckpt_path), f'all_feats_layer_{args.tgt_enc_layer}',f'speech_text_encoder_embeddings_{args.postfix}' + '.pkl'), 'wb') as f:
#         pickle.dump(all_outputs, f)
        
        

# # python dump_embedding.py "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/" --ckpt-path "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/models_diff_bnd_bnd_gradient_DEBUGGING_from_scratch_debug_requires_grad_all_true_noaddenc/checkpoint_best.pt" --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_trial_1/speecht5" --max-text-positions 100 --split "discrere_speech_valid|text_valid"


# # python dump_embedding.py "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/" --ckpt-path "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/models_diff_bnd_without_bnd_gradient_baseline/checkpoint_1_10.pt" --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_trial_1/speecht5" --max-text-positions 100 --split "discrere_speech_valid_true|text_valid_true"


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
    
if __name__ == "__main__":
    import argparse
    import editdistance
    # split, ckpt_path, boundary_path, data_path, layer, pooling_type, feat_dir
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
    dicts["audio"] = Dictionary.load(op.join(args.data, "dict.audio0.txt"))
    speeech_mask_idx = dicts["audio"].add_symbol("<mask>")

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
    all_outputs = []
    print(f'Speech dataset size: {len(all_feats_len)}; Text dataset size: {len(label_list)}')
    with torch.no_grad():
        for idx in range(len(all_feats_len)):
            cur_fn = filenames[idx]
            if idx % 1000 == 0:
                print(f'Processed {idx} samples')
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
                speech_string_list = (speech_meta_info["src_tokens"].cpu().squeeze() - 4).tolist()
                speech_string_list = [str(s) for s in speech_string_list]
                speech_string = " ".join(speech_string_list)
                assert speech_output.size(1) == len(speech_string.split())

            else:
                speech_sample = discrete_labels[idx]
                if src_tokenizer is not None:
                    speech_sample = src_tokenizer.encode(speech_sample)

                if src_label_processor is not None:
                    speech_sample = src_label_processor(speech_sample)

                speech_sample = speech_sample.to("cuda:0").unsqueeze(0)
                speech_encoder_input = model.speech_encoder_prenet.encoder_prenet(speech_sample)
                speech_encoder_padding_mask = torch.BoolTensor(speech_sample.shape).fill_(False).to("cuda:0")

                encoder_output = model.encoder(speech_encoder_input, speech_encoder_padding_mask, tgt_layer=args.tgt_enc_layer)
                speech_output = encoder_output["encoder_out"][0].transpose(0, 1)

                speech_string = discrete_labels[idx]


                assert speech_output.size(1) == len(speech_string.split())
                
            speech_to_text_output = dicts["text"].string(model.text_predictor(speech_output, speech_encoder_padding_mask).argmax(-1).squeeze(0))
                
            all_outputs.append(speech_to_text_output)
            text_sample = label_list[idx]
            
            text_string = text_sample
            gt_text_strings.append(text_string)
            if idx < 10:
                print('GT:', text_string)
                print('Pred:', speech_to_text_output)
            
    print(len(all_outputs), len(filenames), len(gt_text_strings))
    uer_1 = calc_uer_remove_rep(all_outputs, gt_text_strings)
    uer_2 = calc_uer(all_outputs, gt_text_strings)
    print('WER:', uer_1, uer_2)
    os.makedirs(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn_sclite'), exist_ok=True)
    # with open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn_sclite', f'{args.postfix}_preds.txt'), 'w') as f:
    #     for cur_output_pred, cur_output_fn in zip(all_outputs, filenames):
    #         f.write(cur_output_pred + ' (' + os.path.basename(cur_output_fn) + ')'+ '\n')
    with open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn_sclite', f'{args.postfix}_preds.txt'), 'w') as f:
        for cur_output_pred, cur_output_fn in zip(all_outputs, filenames):
            f.write(" ".join([key for key, _group in groupby(cur_output_pred.strip().split())]) + ' (' + os.path.basename(cur_output_fn) + ')'+ '\n')
            
    with open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn_sclite', f'{args.postfix}_ref.txt'), 'w') as f:
        for cur_output_gt, cur_output_fn in zip(gt_text_strings, filenames):
            f.write(cur_output_gt + ' (' + os.path.basename(cur_output_fn) + ')'+ '\n')  
        
        

# python dump_embedding.py "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/" --ckpt-path "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/models_diff_bnd_bnd_gradient_DEBUGGING_from_scratch_debug_requires_grad_all_true_noaddenc/checkpoint_best.pt" --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_trial_1/speecht5" --max-text-positions 100 --split "discrere_speech_valid|text_valid"


# python dump_embedding.py "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/" --ckpt-path "/nobackup/users/junruin2/data/vg_hubert/libri-train-clean/feat_vg_hubert_3_9_ori_feats_gradseg/top_1025/km_dir_1025/models_diff_bnd_without_bnd_gradient_baseline/checkpoint_1_10.pt" --user-dir "/nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_trial_1/speecht5" --max-text-positions 100 --split "discrere_speech_valid_true|text_valid_true"
