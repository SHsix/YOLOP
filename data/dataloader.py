import torch, os
import numpy as np


import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import culane_row_anchor
from data.dataset import LaneTestDataset
from lib.utils import DataLoaderX, torch_distributed_zero_first
import data as data
from .culane import Culane

def get_train_loader(cfg, batch_size, griding_num, dataset, use_aux, distributed, num_lanes):
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((32, 80)),
        mytransforms.MaskToTensor(),
    ])
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    cv_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,    
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(4),
        mytransforms.RandomUDoffsetLABEL(88),
        mytransforms.RandomLROffsetLABEL(160)
    ])

    if dataset == 'CULane':
        train_dataset = Culane(cfg = cfg, is_train= True, inputsize=cfg.MODEL.IMAGE_SIZE, \
            transform = [cv_transform, simu_transform, segment_transform], row_anchor = culane_row_anchor,
                        griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes
            )
        cls_num_per_lane = 18

    else:
        raise NotImplementedError
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, \
        sampler = sampler, num_workers=16, collate_fn=Culane.collate_fn)

    return train_loader, cls_num_per_lane

def get_test_loader(cfg, batch_size, dataset, distributed):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    cv_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,    
    ])
    if dataset == 'CULane':
        test_dataset = Culane(cfg = cfg, is_train= False, inputsize=cfg.MODEL.IMAGE_SIZE, \
        transform = [cv_transform], row_anchor = culane_row_anchor,
                    griding_num=None, use_aux=False, num_lanes = None
        )
        # test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'list/test.txt'),img_transform = cv_transform)

 

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle = False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler, \
        num_workers=16, collate_fn=Culane.test_collate_fn)
    return loader


class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size


        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)