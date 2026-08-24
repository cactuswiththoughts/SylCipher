# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import contextlib

import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


class ExtractedFeaturesDataset(FairseqDataset):
    def __init__(
        self,
        path,
        split,
        min_length=3,
        max_length=None,
        labels=None,
        label_dict=None,
        shuffle=True,
        sort_by_length=True,
        aux_target_postfix=None,
    ):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.label_dict = label_dict
        
        if labels is not None:
            assert label_dict is not None
            
        self.sizes = []
        self.offsets = []
        self.labels = []
        self.aux_tgt = None

        path = os.path.join(path, split)
        data_path = path
        self.data = np.load(data_path + ".npy", mmap_mode="r")
        self.all_feats = np.load(data_path + "_feat.npy", mmap_mode="r")

        offset = 0
        skipped = 0

        if not os.path.exists(path + f".{labels}"):
            labels = None

        self.bad_idxs = []

        bad_files = []
        with open(data_path + '_bnd_errors.txt', 'r') as f:
            bad_lines = f.readlines()

        for bl in bad_lines:
            bad_file = bl.strip().split(maxsplit = 1)[0]
            bad_files.append(bad_file)

        bad_files = set(bad_files)

        self.filenames = []
        with open(data_path + '.files', 'r') as f:
            file_lines = f.readlines()
        for idx, file_line in enumerate(file_lines):
            cur_filename = file_line.strip()
            if cur_filename not in bad_files:
                self.filenames.append(cur_filename)
            else:
                self.bad_idxs.append(idx)
                
        print('fn_len', len(self.filenames))
    

        with open(data_path + ".lengths", "r") as len_f, open(
            path + f".{labels}", "r"
        ) if labels is not None else contextlib.ExitStack() as lbl_f:
            for file_idx, line in enumerate(len_f):
                length = int(line.rstrip())
                lbl = None if labels is None else next(lbl_f).rstrip().split()
                if file_idx not in self.bad_idxs:
                    self.sizes.append(length)
                    self.offsets.append(offset)
                    if lbl is not None:
                        self.labels.append(lbl)
                offset += length

        self.sizes = np.asarray(self.sizes)
        self.offsets = np.asarray(self.offsets)
        
        if aux_target_postfix is not None:
            if not os.path.exists(path+f".{aux_target_postfix}"):
                logger.info(f"auxaliry target for {split} missing")
            else:
                with open(path+f".{aux_target_postfix}", "r") as t_f:
                    self.aux_tgt = [
                        torch.LongTensor(list(map(int,seg.strip().split())))\
                                    for seg in t_f]

        A = np.random.randn(2048, 2048)
        Q, R = np.linalg.qr(A)
        self.centroids = torch.from_numpy(Q.astype(np.float32))

        logger.info(f"loaded {len(self.offsets)}, skipped {skipped} samples")

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.data[offset:end].copy()).float()
        # cont_feats = torch.from_numpy(self.all_feats[offset:end].copy()).float()
        cont_feats = self.centroids[feats.long()]

        if feats.ndim == 1:
            feats = feats.unsqueeze(1)

        res = {"id": index, "features": feats, "cont_features": cont_feats}
        if len(self.labels) > 0:
            res["target"] = self.label_dict.encode_line(
                self.labels[index],
                line_tokenizer=lambda x: x,
                append_eos=False,
            )
        
        if self.aux_tgt:
            res["aux_target"] = self.aux_tgt[index]

        return res

    def __len__(self):
        return len(self.sizes)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        features = [s["features"] for s in samples]
        cont_features = [s["cont_features"] for s in samples]
        sizes = [len(s) for s in features]

        target_size = max(sizes)

        collated_features = features[0].new_zeros(
            len(features), target_size, features[0].size(-1)
        )
        # TODO
        collated_cont_features = cont_features[0].new_zeros(
            len(cont_features), target_size, cont_features[0].size(-1)
        )
        padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
        for i, (f, cf, size) in enumerate(zip(features, cont_features, sizes)):
            collated_features[i, :size] = f
            collated_cont_features[i, :size] = cf
            padding_mask[i, size:] = True
            
        ids = torch.LongTensor([s["id"] for s in samples])
        res = {
            "id": ids,
            "net_input": {"id": ids, "features": collated_features, "cont_features":collated_cont_features, "padding_mask": padding_mask},
        }

        if len(self.labels) > 0:
            target = data_utils.collate_tokens(
                [s["target"] for s in samples],
                pad_idx=self.label_dict.pad(),
                left_pad=False,
            )
            res["target"] = target
        
        if self.aux_tgt:
            idxs = torch.nn.utils.rnn.pad_sequence(
                [s["aux_target"] for s in samples],
                batch_first=True,
                padding_value=-1,
            )
            res["net_input"]["aux_target"] = idxs
        
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        if self.sort_by_length:
            order.append(self.sizes)
            return np.lexsort(order)[::-1]
        else:
            return order[0]
