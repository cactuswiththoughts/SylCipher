# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import random
# from typing import List

# from fairseq.data import BaseWrapperDataset, data_utils


# class RandomInputDataset(BaseWrapperDataset):
#     def __init__(
#         self,
#         dataset,
#         random_input_dataset,
#         input_key_path: List[str],
#         add_to_input,
#     ):
#         super().__init__(dataset)
#         self.random_input_dataset = random_input_dataset
#         if isinstance(input_key_path, str):
#             input_key_path = [input_key_path]
#         assert len(input_key_path) > 0
#         self.input_key_path = input_key_path
#         self.add_to_input = add_to_input

#     def get_target(self, item):
#         target_loc = item
#         for p in self.input_key_path[:-1]:
#             target_loc = target_loc[p]
#         return self.input_key_path[-1], target_loc

#     def get_target_value(self, item):
#         k, target_loc = self.get_target(item)
#         return target_loc[k]

#     def __getitem__(self, index):
#         item = self.dataset[index]
#         k, target_loc = self.get_target(item)
#         target_loc[k] = random.choice(self.random_input_dataset)
#         return item

#     def collater(self, samples):
#         collated = self.dataset.collater(samples)
#         if len(collated) == 0:
#             return collated
#         indices = set(collated["id"].tolist())

#         random_inputs = self.random_input_dataset.collater(
#             [self.get_target_value(s) for s in samples if s["id"] in indices])
        
#         k, target_loc = self.get_target(
#             collated if not self.add_to_input else collated["net_input"]
#         )
#         target_loc[k] = random_inputs
        
#         # print(target_loc.keys())
#         # print(collated.keys())

#         return collated

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List

from fairseq.data import BaseWrapperDataset, data_utils
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RandomInputDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        random_input_dataset,
        input_key_path: List[str],
        add_to_input,
        random_choice=True,
        sample_without_replacement=True
    ):
        super().__init__(dataset)
        self.random_input_dataset = random_input_dataset
        if isinstance(input_key_path, str):
            input_key_path = [input_key_path]
        assert len(input_key_path) > 0
        self.input_key_path = input_key_path
        self.add_to_input = add_to_input
        self.random_choice = random_choice
        self.sample_without_replacement = sample_without_replacement
        # if self.sample_without_replacement:
        #     self.index_mapping = np.random.permutation(len(dataset))
        # else:
        #     self.index_mapping = np.random.randint(low=0, high=len(dataset), size=len(dataset))
        if self.sample_without_replacement:
            self.index_mapping = np.random.choice(a = len(self.random_input_dataset), size=len(self.dataset), replace=False)
        else:
            self.index_mapping = np.random.randint(low=0, high=len(self.random_input_dataset), size=len(self.dataset))
        
        logger.info(f'A new instance of RandomInputDataset has been created!')

    def get_target(self, item):
        target_loc = item
        for p in self.input_key_path[:-1]:
            target_loc = target_loc[p]
        return self.input_key_path[-1], target_loc

    def get_target_value(self, item):
        k, target_loc = self.get_target(item)
        return target_loc[k]

    def __getitem__(self, index):
        item = self.dataset[index]
        k, target_loc = self.get_target(item)
        if self.random_choice:
            target_loc[k] = self.random_input_dataset[self.index_mapping[index]]
        else:
            target_loc[k] = self.random_input_dataset[index]
        return item

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())

        random_inputs = self.random_input_dataset.collater(
            [self.get_target_value(s) for s in samples if s["id"] in indices])
        
        k, target_loc = self.get_target(
            collated if not self.add_to_input else collated["net_input"]
        )
        target_loc[k] = random_inputs
        
        # print(target_loc.keys())
        # print(collated.keys())

        return collated
    
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        logger.info(f'Setting epoch {epoch}')
        
        if (int(epoch) % 2) == 1:
            if self.sample_without_replacement:
                self.index_mapping = np.random.choice(a = len(self.random_input_dataset), size=len(self.dataset), replace=False)
            else:
                self.index_mapping = np.random.randint(low=0, high=len(self.random_input_dataset), size=len(self.dataset))
                
            logger.info(f'A new instance of RandomInputDataset has been created in set_epoch!')
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)