import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from .mlr_viper import MLRViper
from .mlr_market1501 import MLRMarket1501
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
__factory = {
    'mlr_viper': MLRViper,
    'mlr_market1501': MLRMarket1501
}

def train_collate_fn(batch):
    imgs, hr_imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(hr_imgs, dim=0), pids, camids, viewids,

def train_collate_fn_single(batch):
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(name='mlr_viper', bs=32, root_dir='./data'):

    val_transforms = T.Compose([
        T.Resize([384, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_workers = 4
    dataset_name = name
    data_root_dir = root_dir

    dataset = __factory[dataset_name](root=data_root_dir)


    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    query_set = ImageDataset(dataset.query, val_transforms)
    test_set = ImageDataset(dataset.gallery, val_transforms)


    query_loader = DataLoader(
        query_set, batch_size=bs, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=bs, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return query_loader, test_loader, len(dataset.query), num_classes, cam_num, view_num
