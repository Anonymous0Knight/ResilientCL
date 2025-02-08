from torch.utils.data import Dataset as dataset
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms


from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


def get_csv(file, delimiter=','):
    import csv
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        result = list(reader)
    return result


## get json content as dict
def get_json(file):
    import json
    with open(file, 'r', encoding='utf-8') as f:
        dicts = json.load(f)
    return dicts


class ImagenetLTDataset(dataset):
    def __init__(self, dire, txt, transform=None):
        super(ImagenetLTDataset, self).__init__()

        self.dire = dire
        self.data = get_csv(txt, delimiter=' ')
        self.labels = list(map(int, np.asarray(self.data)[:, 1]))
        self.transform = transform

        self.imsize = len(self.data)

    def __len__(self):
        return self.imsize

    def __getitem__(self, index):
        path, label = self.data[index]

        with open(os.path.join(self.dire, path), 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


class iNat18Dataset(dataset):
    def __init__(self, dire, txt, transform=None):
        super(iNat18Dataset, self).__init__()

        self.dire = dire
        self.data =  get_json(txt)
        self.transform = transform

        self.imsize = len(self.data['images'])

    def __len__(self):
        return self.imsize

    def __getitem__(self, index):
        path = self.data['images'][index]['file_name']
        label = self.data['annotations'][index]['category_id']

        with open(os.path.join(self.dire, path), 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)



