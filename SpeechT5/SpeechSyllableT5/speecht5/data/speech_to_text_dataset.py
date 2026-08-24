# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import itertools
import logging
import os
from typing import Any, List, Optional

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils, Dictionary
from fairseq.data.fairseq_dataset import FairseqDataset
import pickle
import json
import sys

logger = logging.getLogger(__name__)




# def load_label(label_path, file_list):
    
#     with open(label_path, 'r') as f:
#         label_json = json.load(f)['data']
    
#     wav2trans = {}
#     for item in label_json:
#         wav_fn = item["caption"]["wav"]
#         wav_trans = item["caption"]["text"]
#         wav2trans[wav_fn] = wav_trans
    
#     labels = []
        
#     for file in file_list:
#         labels.append(wav2trans[file])
        
#     return labels



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



class SpeechToTextDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        boundary_path: Optional[str] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        reduction_factor: int = 1,
        tgt_dict: Optional[Dictionary] = None,
        tokenizer = None,
        label_processors: Optional[List[Any]] = None,
        src_dict: Optional[Dictionary] = None,
        src_tokenizer = None,
        src_label_processors: Optional[List[Any]] = None,
    ):
        self.src_dict = src_dict
        self.src_tokenizer = src_tokenizer
        self.src_label_processors = src_label_processors[0]
        
        self.discrete_labels = []
        
        self.bad_idxs = []

        bad_files = []
        with open(os.path.join(manifest_path, 'bnd_errors.txt'), 'r') as f:
            bad_lines = f.readlines()

        for bl in bad_lines:
            bad_file = bl.strip().split(maxsplit = 1)[0]
            bad_files.append(bad_file)

        bad_files = set(bad_files)

        self.filenames = []
        with open(os.path.join(manifest_path, 'file_list.txt'), 'r') as f:
            file_lines = f.readlines()
        for idx, file_line in enumerate(file_lines):
            cur_filename = file_line.strip()
            if cur_filename not in bad_files:
                self.filenames.append(cur_filename)
            else:
                self.bad_idxs.append(idx)
    
        # self.bad_idxs = set(self.bad_idxs)

        remove_file_idx = []
        file_idx = 0
        with open(os.path.join(manifest_path, 'data.lengths'), 'r') as f:
            length_lines = f.readlines()
        for idx, len_line in enumerate(length_lines):
            # print(int(len_line.strip()))
            if idx not in self.bad_idxs:
                if int(len_line.strip()) > 20000:
                    self.bad_idxs.append(idx)
                    remove_file_idx.append(file_idx)
                file_idx += 1

        remove_file_idx = set(remove_file_idx)
        self.filenames = [self.filenames[idx] for idx in range(len(self.filenames)) if idx not in remove_file_idx]


        self.sizes = []
        self.text_dict_symbols = None
        if tgt_dict is not None:
            self.text_dict_symbols = set(tgt_dict.symbols)

        exceed_length_idxs = []
        remove_file_idx = []
        file_idx = 0
        with open(os.path.join(manifest_path, 'data.km')) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx not in self.bad_idxs:
                    if len(line.strip().split()) <= 300:
                        self.discrete_labels.append(line.strip())
                        self.sizes.append(len(line.strip().split()))
                    else:
                        exceed_length_idxs.append(idx)
                        remove_file_idx.append(file_idx)
                    file_idx += 1

        logger.info(f'Max size of audio:, {max(self.sizes)}')
        self.bad_idxs.extend(exceed_length_idxs)
        self.bad_idxs = set(self.bad_idxs)
        remove_file_idx = set(remove_file_idx)
        self.filenames = [self.filenames[idx] for idx in range(len(self.filenames)) if idx not in remove_file_idx]
                          
                
        self.all_feats = np.load(os.path.join(manifest_path, 'data.npy'),mmap_mode='r')
        # self.all_feats_len = []
        # with open(os.path.join(manifest_path, 'data.lengths'), 'r') as f:
        #     length_lines = f.readlines()
        # for len_line in length_lines:
        #     # print(int(len_line.strip()))
        #     self.all_feats_len.append(int(len_line.strip()))

        # self.offset = [0] + list(np.cumsum(self.all_feats_len)[:-1])

        # new_offset = []
        # new_all_feats_len = []
        # for idx in range(len(self.offset)):
        #     if idx not in self.bad_idxs:
        #         new_offset.append(self.offset[idx])
        
        # for idx in range(len(self.all_feats_len)):
        #     if idx not in self.bad_idxs:
        #         new_all_feats_len.append(self.all_feats_len[idx])

        # self.offset = new_offset
        # self.all_feats_len = new_all_feats_len

        offset = 0
        self.all_feats_len = []
        self.offset = []
        with open(os.path.join(manifest_path, 'data.lengths'), 'r')as len_f:
            for idx, line in enumerate(len_f):
                length = int(line.strip())

                if idx not in self.bad_idxs:
                    self.all_feats_len.append(length)
                    self.offset.append(offset)
                offset += length


        # self.sizes = np.array(self.all_feats_len)
        logger.info(f'Max size of audio feats:, {max(self.all_feats_len)}')
            
        with open(os.path.join(manifest_path, 'data_dict_bnd.pkl'), 'rb') as fb:
            boundary_dict = pickle.load(fb)
        
        self.boundaries = []
        # for idx, filename in enumerate(self.filenames):
        #     if self.all_feats_len[idx] == 0:
        #         print('bad file encountered')
        #     boundary_arr = np.zeros(self.all_feats_len[idx]).astype('int32')
        #     boundaries = boundary_dict[filename]["boundaries"]
        #     for boundary_seg in boundaries:
        #         boundary_arr[int(50*boundary_seg[0]): int(50 * boundary_seg[1])] = 1
        #     self.boundaries.append(boundary_arr)
        
        for idx, filename in enumerate(self.filenames):
            boundary_arr = np.zeros(self.all_feats_len[idx]).astype('int32')
            boundaries = boundary_dict[filename]
            
            boundary_locs = np.where(boundaries['seg_bound'] == 1)[0]
            # print(boundary_locs, len(curr_data))
            if len(boundary_locs) == 0:
                boundary_locs = np.arange(0, 2 * (self.all_feats_len[idx]), 26).astype('int32')
                # print('empty boundaries found!')
            if len(boundary_locs) == 1:
                boundary_locs = np.arange(0, 2 * (self.all_feats_len[idx]), 26).astype('int32')
                # print('length-one boundaries found!')
            if len(boundary_locs) == 1:
                boundary_locs = [0, 2 * (self.all_feats_len[idx]) - 1]
                boundary_locs = np.array(boundary_locs).astype('int32')
                # print('length-one boundaries found but due to short utt!')

            # for start_frame2x, end_frame2x in zip(boundary_locs[:-1], boundary_locs[1:]):
            #     start_frame = start_frame2x // 2
            #     end_frame = end_frame2x // 2
            #     # print(start_frame, end_frame, len(curr_data))
            #     if start_frame != end_frame:
            #         curr_seg_data = np.mean(curr_data[start_frame: end_frame, :], axis = 0)
            #         curr_seg_feats.append(curr_seg_data)
            #     else:
            #         curr_seg_data = np.mean(curr_data[start_frame: end_frame + 1, :], axis = 0)
            #         curr_seg_feats.append(curr_seg_data)
            boundary_locs = np.minimum(boundary_locs, 2 * self.all_feats_len[idx] - 1)            
            boundary_arr[boundary_locs // 2] = 1
            # boundary_arr[0] = 1
            # boundary_arr[-1] = 1
            self.boundaries.append(boundary_arr)
            
        # XXX
        # self.filenames = self.filenames[:100]
        # self.sizes = self.sizes[:100] 
        # self.offset = self.offset[:100]
        # self.all_feats_len = self.all_feats_len[:100]  
        # self.discrete_labels = self.discrete_labels[:100] # XXX
        
        self.label_list = [load_label(os.path.join(manifest_path, 'meta_data.json'), self.filenames, self.text_dict_symbols)]
        
        logger.info(f"S2T DATALOADER LENGTH CHECK: {len(self.offset)}, {len(self.all_feats_len)}, {len(self.boundaries)}, {len(self.filenames), len(self.discrete_labels)}, {len(self.label_list)}")
        
        self.shuffle = shuffle
        self.num_labels = 1
        self.single_target = True
        

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        # print('see here', self.max_sample_size)
        self.pad_audio = pad_audio

        self.reduction_factor = reduction_factor
        
        self.tgt_dict = tgt_dict
        self.tokenizer = tokenizer
        self.label_processors = label_processors
        
        logger.info(
            f"pad_audio={pad_audio}, reduction_factor={reduction_factor}, "
            f"max_sample_size={self.max_sample_size}"
        )

    def get_audio(self, index):
        wav = self.all_feats[self.offset[index]: self.offset[index] + self.all_feats_len[index]]
        boundaries = self.boundaries[index]
        # print(len(wav), len(label))
        # if len(wav) > self.max_sample_size:
        #     wav = wav[:self.max_sample_size]
        #     boundaries = boundaries[:self.max_sample_size]
        return torch.from_numpy(wav), torch.from_numpy(boundaries)

    def get_label(self, index, label_idx):
        label = self.label_list[label_idx][index]

        if self.tokenizer is not None:
            label = self.tokenizer.encode(label)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav, boundaries = self.get_audio(index)
        labels = self.get_labels(index)
    
        
        tokens = self.discrete_labels[index]
        
        # print('1', tokens)
        if self.src_tokenizer is not None:
            tokens = self.src_tokenizer.encode(tokens)

        if self.src_label_processors is not None:
            tokens = self.src_label_processors(tokens)
            
        # print('2', tokens)
        
        return {"id": index, "source": wav, "boundaries": boundaries, "label_list": labels, "tokens": tokens}
    
    def __len__(self):
        return len(self.all_feats_len)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}
        
        audio_tokens = [s["tokens"] for s in samples]

        audio_token_sizes = [len(s) for s in audio_tokens]
        # print('audio', sizes)
        audio_token_size = max(audio_token_sizes)

        audios_by_label = [audio_tokens]
        audios_list, audio_lengths_list, audio_ntokens_list = self.collater_label(audios_by_label, self.src_dict)
        # audios_list =[torch.cat((torch.tensor([self.src_dict.bos()]), audios_list[0][i, :audio_lengths_list[0][i]].long(), torch.tensor([self.src_dict.eos()])), 0) for i in range(audios_list[0].size(0))]

        audios_list_noboseos = [audios_list[0][i, :audio_lengths_list[0][i]].long()for i in range(audios_list[0].size(0))]

        collated_audio_tokens = data_utils.collate_tokens(
            audios_list_noboseos,
            self.src_dict.pad(),
            self.src_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        
        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]

        boundaries = [s["boundaries"] for s in samples]
        #boundary_sizes = [len(s) for s in boundaries]

        collated_audios, padding_mask, audio_size = self.collater_audio(
            audios)
        collated_audios_size = torch.tensor([audio_size for audio_size in audio_sizes], dtype=torch.long)

        collated_boundaries, collated_boundaries_lengths, collated_boundaries_ntokens = self.collater_boundary_label(
            boundaries, pad=-1
        )
        
        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(targets_by_label, self.tgt_dict)

        decoder_label = [
            torch.cat((targets_list[0][i, :lengths_list[0][i]], torch.tensor([self.tgt_dict.eos()])), 0).long()
            for i in range(targets_list[0].size(0))
        ]

        decoder_target = data_utils.collate_tokens(
            decoder_label,
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        decoder_target_lengths = torch.tensor(
            [x.size(0) for x in decoder_label], dtype=torch.long
        )
        prev_output_tokens = data_utils.collate_tokens(
            decoder_label,
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )

        net_input = {
            "source": collated_audios,
            "audio_tokens": collated_audio_tokens,
            "padding_mask": padding_mask,
            "boundaries": collated_boundaries,
            "prev_output_tokens": prev_output_tokens,
            "task_name": "s2t",
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "target": decoder_target,
            "target_lengths": decoder_target_lengths,
            "task_name": "s2t",
            "ntokens": ntokens_list[0]
        }
        # print(ntokens_list[0])
        return batch
    
    def collater_audio(self, features):
        sizes = [len(s) for s in features]
        # print('audio', sizes)
        target_size = max(sizes)

        collated_features = features[0].new_zeros(
            len(features), target_size, features[0].size(-1)
        )
        padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
        for i, (f, size) in enumerate(zip(features, sizes)):
            real_size = size
            collated_features[i, :real_size] = f[:real_size]
            padding_mask[i, size:] = True

        return collated_features, padding_mask, target_size
    
    def collater_boundary_label(self, targets, pad=-1):
        # print(targets)
        lengths = torch.LongTensor([len(t) for t in targets])
        # print(targets)
        ntokens = lengths.sum().item()
        # print(lengths, len(targets), torch.max(lengths))
        collated_targets = torch.LongTensor(len(targets), torch.max(lengths))
        for idx, t in enumerate(targets):
            collated_targets[idx, :lengths[idx]] = t
            collated_targets[idx, lengths[idx]: ] = pad
        return collated_targets, lengths, ntokens    


    def collater_seq_label(self, targets, pad):
        # print(targets)
        lengths = torch.LongTensor([len(t) for t in targets])
        
        targets = [targets[idx][:int(lengths[idx])] for idx in range(len(lengths))]

        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, dictionary):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, [dictionary.pad()])
        for targets, pad in itr:
            targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list
    
    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]


    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
