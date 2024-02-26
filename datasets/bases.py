from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import logging
import numpy as np
import torchvision.transforms as T
from utils.image_tools import Addblur, AddGaussianNoise, AddSaltPepperNoise
# from config import cfg  # import defaults.py as cfg
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data, fintune=False):
        pids, cams, tracks = [], [], []
        if not fintune:
            for _, pid, camid, trackid in data:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
        else:
            for _, _, pid, camid, trackid in data:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery, fintune=False):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train, fintune=fintune)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        logger.info("  ----------------------------------------")

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path
        #  return img, pid, camid, trackid,img_path.split('/')[-1]

class MLRImageDataset(Dataset):
    def __init__(self, dataset, transform=None, fintune=False, data_name=''):
        self.dataset = dataset
        self.transform = transform
        self.mlr_dataset = fintune
        self.name = data_name
        self.transform_noise = T.Compose([
            T.Resize([384, 128], interpolation=3),  # cfg.INPUT.SIZE_TRAIN
            Addblur(p=0.3, blur="Gaussian"),
            # 注意要加这两个东西
            T.ToTensor(),
            T.ToPILImage(),
            AddSaltPepperNoise(0.01, 0.3),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not self.mlr_dataset:
            img_path, pid, camid, trackid = self.dataset[index]
            if 'mlr_' in img_path:
                hr_img_path = img_path.replace('mlr_', '')
            else:
                hr_img_path = img_path
        else:
            hr_img_path, img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        hr_img = read_image(hr_img_path)

        # 用该方法获取一个随机种子
        seed = torch.random.seed()
        seed_py = np.random.randint(2147483647)
        if self.transform is not None:
            # if 'mlr_caviar' in self.name:
            #     img = self.transform_noise(img)
            # 该方法设置随机种子
            torch.random.manual_seed(seed)
            random.seed(seed_py)
            img = self.transform(img)
            # 该方法设置随机种子
            torch.random.manual_seed(seed)
            random.seed(seed_py)
            hr_img = self.transform(hr_img)

        return img, hr_img, pid, camid, trackid, img_path
