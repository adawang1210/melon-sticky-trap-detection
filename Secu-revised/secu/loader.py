# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# modify from
# https://github.com/facebookresearch/moco-v3/blob/main/moco/loader.py

from PIL import Image, ImageFilter, ImageOps
import random


class SingleCropsTransform:
    """Take a single random crop of one image"""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x)

class DoubleCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class MultiCropsTransform:
    """Take multiple random crops of one image"""

    def __init__(self, base_transform1, base_transform2, small_transform, snum):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.small_transform = small_transform
        self.snum = snum

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        simgs = []
        for i in range(0, self.snum):
            simgs.append(self.small_transform(x))
        return [im1, im2, simgs]


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
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std
    
# ---------- 銳化工具 ----------
class RandomUnsharpMask:
    def __init__(self, p=0.5, radius=(1.0, 2.0), percent=(100, 150), threshold=3):
        self.p, self.radius, self.percent, self.threshold = p, radius, percent, threshold
    def __call__(self, img):
        if random.random() < self.p:
            r = random.uniform(*self.radius)
            p = random.randint(*self.percent)
            return img.filter(ImageFilter.UnsharpMask(radius=r, percent=p, threshold=self.threshold))
        return img

# ---------- JPEG 擾動 ----------
class RandomJPEGCompression:
    def __init__(self, p=0.5, quality=(60, 90)):
        self.p, self.quality = p, quality
    def __call__(self, img):
        if random.random() < self.p:
            from io import BytesIO
            buf = BytesIO()
            q = random.randint(*self.quality)
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img