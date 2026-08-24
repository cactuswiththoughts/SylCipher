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
import sys
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
import librosa
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform
from fairseq.data import data_utils, Dictionary
from fairseq.data.fairseq_dataset import FairseqDataset
import pickle
import math

logger = logging.getLogger(__name__)

def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
):
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out

def add_first_frame_and_remove_last_frame(ys):
    ys_in = torch.cat(
        [ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1
    )
    return ys_in

def collate(
    source,
    unmasked_source,
    targets,
    pad_idx,
    task_name = 'speech_text_pretrain',
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    assert input_feeding
    if len(source) == 0:
        return {}
    
    no_boseos_targets = [s[1:-1] for s in targets]

    def merge(input, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            input,
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )
    src_tokens = merge(
        source,
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    src_lengths = torch.LongTensor([s.numel() for s in source])

    prev_output_tokens = None
    target = None
    target = merge(
        targets,
        left_pad=left_pad_target,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )

    no_boseos_target = merge(
        no_boseos_targets,
        left_pad=left_pad_target,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )
    ntokens = sum([len(cur_target) for cur_target in targets])

    if input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            targets,
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
 
    unmasked_sizes = [len(s) for s in unmasked_source]

    unmasked_target_size = max(unmasked_sizes)

    unmasked_src_tokens = unmasked_source[0].new_zeros(
        len(unmasked_source), unmasked_target_size
    )
    for i, (f, size) in enumerate(zip(unmasked_source, unmasked_sizes)):
        unmasked_src_tokens[i, :size] = f

    unmasked_src_lengths = torch.LongTensor([s.numel() for s in unmasked_source])
    
    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "unmasked_src_tokens": unmasked_src_tokens,
            "unmasked_src_lengths": unmasked_src_lengths,
        },
        "target": target,
        "no_boseos_target": no_boseos_target,
        "nsentences": len(source),
        "task_name": task_name,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch


class SpeechPretrainDataset(FairseqDataset):
    def __init__(
        self,
        mask_whole_word, 
        mask_idx, 
        manifest_path: str,
        boundary_path: Optional[str] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        reduction_factor: int = 1,
        src_dict: Optional[Dictionary] = None,
        src_tokenizer = None,
        src_label_processors: Optional[List[Any]] = None,
        mask_length = "span-poisson", 
        poisson_lambda = 3.5,
        mask = 0.3, 
        mask_random = 0.1, 
        iid_noise_target = False, 
        replace_length=1, 
        uni_mask_idxs = None
    ):
        self.src_dict = src_dict
        self.vocab = src_dict
        self.mask_whole_word = mask_whole_word
        self.mask_idx = mask_idx
        self.padding_idx = self.vocab.pad()
        self.random_ratio = mask_random
        self.mask_ratio = mask
        self.mask_span_distribution = None
        
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
            self.mask_span_distribution = torch.distributions.Categorical(ps)
            
        self.iid_noise_target = iid_noise_target
        self.uni_mask_idxs = uni_mask_idxs
        self.replace_length = replace_length
        
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
    
        self.bad_idxs = set(self.bad_idxs)
        
        self.sizes = []

        with open(os.path.join(manifest_path, 'data.km')) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx not in self.bad_idxs:
                    self.discrete_labels.append(line.strip())
                    self.sizes.append(len(line.strip().split()))
                    
                    
        self.frame_clus_labels = []
        with open(os.path.join(manifest_path, 'data_frame_clus.km')) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx not in self.bad_idxs:
                    self.frame_clus_labels.append(np.array([int(a) for a in line.strip().split()]))


        self.all_feats = np.load(os.path.join(manifest_path, 'data.npy'), mmap_mode='r')
        self.all_feats_len = []
        with open(os.path.join(manifest_path, 'data.lengths'), 'r') as f:
            length_lines = f.readlines()
        for len_line in length_lines:
            # print(int(len_line.strip()))
            self.all_feats_len.append(int(len_line.strip()))

        self.offset = [0] + list(np.cumsum(self.all_feats_len)[:-1])

        new_offset = []
        new_all_feats_len = []
        for idx in range(len(self.offset)):
            if idx not in self.bad_idxs:
                new_offset.append(self.offset[idx])
        
        for idx in range(len(self.all_feats_len)):
            if idx not in self.bad_idxs:
                new_all_feats_len.append(self.all_feats_len[idx])

        self.offset = new_offset
        self.all_feats_len = new_all_feats_len
        # XXX
        # self.discrete_labels = self.discrete_labels[:400000]
        # self.frame_clus_labels = self.frame_clus_labels[:400000]
        # self.sizes = self.sizes[:400000]
        # self.offset = self.offset[:400000] 
        # self.filenames = self.filenames[:400000]
        # self.all_feats_len = new_all_feats_len[:400000] # XXX

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

            if boundary_locs[-1] > 2 * self.all_feats_len[idx]: 
                print(f'{boundary_locs} > {2 * self.all_feats_len[idx]}')     
            boundary_locs = np.minimum(boundary_locs, 2 * self.all_feats_len[idx] - 1)
            
            boundary_arr[boundary_locs // 2] = 1
            boundary_arr[0] = 1
            # boundary_arr[-1] = 1
            self.boundaries.append(boundary_arr)

        self.shuffle = shuffle
        self.num_labels = 1
        self.single_target = True


        logger.info(f"SPEECH DATALOADER LENGTH CHECK: {len(self.offset)}, {len(self.all_feats_len)}, {len(self.boundaries)}, {len(self.filenames), len(self.discrete_labels)}")

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        # print('see here', self.max_sample_size)
        self.pad_audio = pad_audio

        self.reduction_factor = reduction_factor
        logger.info(
            f"pad_audio={pad_audio}, reduction_factor={reduction_factor}, "
            f"max_sample_size={self.max_sample_size}"
        )

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start
    
    def add_whole_word_mask(self, source, p):
        source_ori = source.clone()
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
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
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                1, len(self.vocab), size=(mask_random.sum(),)
            ).to(source.device)

        if self.mask_span_distribution is not None:
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
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    ).to(source.device)
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    ).to(source.device)

                assert source_length - 1 not in indices

        if not self.iid_noise_target:
            unmasked_source = source_ori[to_keep]
            source = source[to_keep]
            target = None
            assert len(unmasked_source) == len(source)
        else:
            print('Warning! iid_noise_target is not False')
            ## Prepare source
            source_mask_idx = (source == self.mask_idx).nonzero().view(-1)
            source[source_mask_idx] = self.uni_mask_idxs[:source_mask_idx.size(0)]
            source = source[to_keep]

            ## Prepare target
            to_keep[source_mask_idx] = 0

            # source_mask_idx: from [a, b, c, ...] to [a, b + 1, c + 2, ...]
            source_mask_idx = source_mask_idx + torch.arange(source_mask_idx.size(0))
            # target: source_length + mask_length
            target = source_ori.new_zeros(source_mask_idx.size(0) + source_ori.size(0))
            # target: [0, 0, 0, X, 0, 0, Y, ....]
            target[source_mask_idx] = self.uni_mask_idxs[:source_mask_idx.size(0)]

            target_to_keep = to_keep.new_zeros(source_mask_idx.size(0) + source_ori.size(0))

            # Copy original value to target and target_to_keep
            target_to_keep[target == 0] = to_keep
            target_to_keep[-1] = 0
            target[target == 0] = source_ori

            target = target[~target_to_keep]

        # if num_inserts > 0:
        #     source, unmasked_source = self.add_insertion_noise(source, unmasked_source, num_inserts / source.size(0))

        return source, unmasked_source, target
    
    def add_insertion_noise(self, tokens, unmasked_tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1).to(tokens.device)
        unmasked_result = torch.LongTensor(n + len(tokens)).fill_(-1).to(tokens.device)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=len(self.vocab), size=(num_random,)
        ).to(tokens.device)
        unmasked_result[noise_indices[num_random:]] = result[noise_indices[num_random:]].clone()
        unmasked_result[noise_indices[:num_random]] = result[noise_indices[:num_random]].clone()
        

        result[~noise_mask] = tokens
        unmasked_result[~noise_mask] = unmasked_tokens

        assert (result >= 0).all()
        return result, unmasked_result

    def get_audio(self, index):
        # print(self.offset[index], self.all_feats_len[index])
        wav = self.all_feats[self.offset[index]: self.offset[index] + self.all_feats_len[index]]
        label = self.boundaries[index]
        aux_frame_clus = self.frame_clus_labels[index]
        # print(len(wav), len(label))
        # if len(wav) > self.max_sample_size:
        #     wav = wav[:self.max_sample_size]
        #     label = label[:self.max_sample_size]
        
        
        return torch.from_numpy(wav), torch.from_numpy(label).long(), torch.from_numpy(aux_frame_clus).long()

    def __getitem__(self, index):
        if index == 0:
            print(self.filenames[index])
        wav, labels, aux_frame_clus = self.get_audio(index)
        
        tokens = self.discrete_labels[index]
        # print('1', tokens)
        if self.src_tokenizer is not None:
            tokens = self.src_tokenizer.encode(tokens)

        if self.src_label_processors is not None:
            tokens = self.src_label_processors(tokens)
        
        unmasked_source = tokens.clone()
        
        tokens = torch.cat([torch.LongTensor([self.vocab.bos()]),
                            tokens,
                            torch.LongTensor([self.vocab.eos()])])
        
        # print('2', tokens)
            
        source, target = tokens, tokens.clone()
        if self.mask_ratio > 0:
            source, _, new_target = self.add_whole_word_mask(source, self.mask_ratio)
            # assert len(source) == len(unmasked_source)
            if new_target is not None:
                target = new_target
            
        return {
            "id": index,
            "source": wav,
            "boundaries": labels,
            "aux_frame_clus": aux_frame_clus,
            "tokens": source,
            "unmasked_tokens": unmasked_source,
            "tokens_target": target,
        }

    def __len__(self):
        return len(self.all_feats_len)

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]

        boundaries = [s["boundaries"] for s in samples]
        aux_frame_clus = [s["aux_frame_clus"] for s in samples]
        #boundary_sizes = [len(s) for s in boundaries]
        
        for b, a in zip(boundaries, aux_frame_clus):
            assert len(b) == len(a)

        collated_audios, padding_mask, audio_size = self.collater_audio(
            audios)
        collated_audios_size = torch.tensor([audio_size for audio_size in audio_sizes], dtype=torch.long)

        targets, lengths, ntokens = self.collater_seq_label(
            boundaries, pad=0
        )
        aux_frame_clus_targets, aux_frame_clus_lengths, aux_frame_clus_ntokens = self.collater_seq_label(
            aux_frame_clus, pad=0
        )
        
        tokens = [s["tokens"] for s in samples]
        unmasked_tokens = [s["unmasked_tokens"] for s in samples]
        tokens_target = [s["tokens_target"] for s in samples]
        
        discrete_batch = collate(tokens, unmasked_tokens, tokens_target, self.padding_idx) 

        net_input = {
            "source": collated_audios, 
            "padding_mask": padding_mask,
            "boundaries": targets,
            "aux_frame_clus":aux_frame_clus_targets,
            "discrete_batch": discrete_batch,
        }

        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "src_lengths": collated_audios_size,
            "task_name": 'speech_pretrain',
        }
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

    def collater_seq_label(self, targets, pad=-1):
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

    def num_tokens(self, index):
        return self.size(index, use_raw=True)

    def size(self, index, use_raw=False):
        if use_raw:
            if self.pad_audio:
                return self.sizes[index]
            real_index = index
            if self.pad_audio:
                return self.sizes[real_index]
            return min(self.sizes[real_index], self.max_sample_size)
        else:
            real_index = index
            if self.pad_audio:
                return self.sizes[real_index]
            return min(self.sizes[real_index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append([self.sizes[i] for i in range(len(self.sizes))])
        return np.lexsort(order)[::-1]
