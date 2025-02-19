# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from hashlib import md5
import torch
import numpy as np
from torch.utils.data import Dataset

def pad_1d_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    ):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

def pad_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    ):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, size - len(v) :] if left_pad else res[i][: len(v), : len(v)])
    return res

def pad_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    ):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v),:])
    return res

def batch_collate_fn(batch):
    coord =[torch.tensor(i[0]) for i in batch]
    atype =[np.array(i[1]) for i in batch]
    target = torch.tensor([torch.tensor(i[2]) for i in batch])
    return coord,atype,target.float()

def batch_collate_fn_infer(batch):
    coord =[torch.tensor(i[0]) for i in batch]
    atype =[np.array(i[1]) for i in batch]
    return coord,atype,


def batch_collate_fn_index(batch):
    coord =[torch.tensor(i[0]) for i in batch]
    atype =[np.array(i[1]) for i in batch]
    index =[np.array(i[3]) for i in batch]
    return coord,atype,index

def batch_collate_fn_uniput(batch):
    input_dict ={
        'src_coord' : torch.stack([i[0] for i in batch]),
        'src_distance' : torch.stack([i[1] for i in batch]),
        'src_edge_type' : torch.stack([i[2] for i in batch]),
        'src_tokens' : torch.stack([i[3] for i in batch])
        }
    target =torch.tensor([[i[4]] for i in batch])
    return input_dict,target.float()

class unimol_inputs_dataset(Dataset):
    def __init__(self,src_coord,src_distance,src_edge_type,src_tokens,target):
        self.src_coord = src_coord
        self.src_distance = src_distance
        self.src_edge_type = src_edge_type
        self.src_tokens = src_tokens
        self.target = target
    
    def __len__(self):
        return len(self.src_coord)
    
    def __getitem__(self,idx):
        return self.src_coord[idx],self.src_distance[idx],self.src_edge_type[idx],self.src_tokens[idx],self.target[idx]