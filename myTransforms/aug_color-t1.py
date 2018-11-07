#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import random

imagenet_pca = {"eigval": torch.Tensor([0.2175, 0.0188, 0.0045]),
                    "eigvec": torch.Tensor([
                        [-0.5675,  0.7192,  0.4009],
                        [-0.5808, -0.0045, -0.8140],
                        [-0.5836, -0.6948,  0.4203],
                    ])
                }

class ToTensor_numpy():

    def __init__(self, channel=6):
        self.channel = channel

    def __call__(self, img):
        channel = img.shape[2]
        assert channel >= self.channel
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.copy()) # .cuda()
        img[:self.channel] /= 255.0
        return img

class Normalize():

    def __init__(self, mean, std, group=1):
        assert (len(mean) == 3) and (len(std) == 3)
        self.group = group
        self.mean = mean
        self.std = std

    def __call__(self, img):
        group = min(self.group, img.shape[0]//3)
        for grp in range(group):
            idx = grp*3
            for i in range(3):
                if(self.mean[i] != 0):
                    img[idx + i] -= self.mean[i]
                if(self.std[i] != 1):
                    img[idx + i] /= self.std[i]
        return img

class UnNormalize():

    def __init__(self, mean, std, group=1):
        assert (len(mean) == 3) and (len(std) == 3)
        self.group = group
        self.mean = mean
        self.std = std

    def __call__(self, img):
        group = min(self.group, img.shape[0]//3)
        for grp in range(group):
            idx = grp*3
            for i in range(3):
                if(self.std[i] != 1):
                    img[idx + i] *= self.std[i]
                if(self.mean[i] != 0):
                    img[idx + i] += self.mean[i]
        return img

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd=0.1, group=1, same_group=False):
        self.alphastd = alphastd
        self.eigval = imagenet_pca["eigval"]
        self.eigvec = imagenet_pca["eigvec"]
        self.group = group
        self.same_group = same_group

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        group = min(self.group, img.shape[0]//3)
        same_group = self.same_group and (group>1)
        if(same_group):
            alpha = img[:3].new().resize_(3).normal_(0, self.alphastd)
            for grp in range(group):
                idx = 3*grp
                img[idx:idx+3] = self.__call_rgb__(img[idx:idx+3], alpha)
        else:
            for grp in range(group):
                alpha = img[:3].new().resize_(3).normal_(0, self.alphastd)
                img[idx:idx+3] = self.__call_rgb__(img[idx:idx+3], alpha)
        return img
        
    def __call_rgb__(self, img, alpha):
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.type_as(img).view(1, 3).expand(3, 3))\
            .mul(self.eigval.type_as(img).view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img)).clamp(0, 1)
        


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        group = img.shape[0]//3
        for grp in range(group):
            idx = grp * 3
            idx1 = idx + 1
            idx2 = idx + 2
            gs[idx].mul_(0.299).add_(0.587, gs[idx1]).add_(0.114, gs[idx2])
            gs[idx1].copy_(gs[idx])
            gs[idx2].copy_(gs[idx])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        group = img.shape[0]//3
        alpha = [random.uniform(-0.5, 0.5)*self.var for i in range(3)]
        for grp in range(group):
            idx = grp*3
            for i in range(3):
                img[idx + i] = img[idx + i] + gs[idx + i]*alpha[0]
        return img

class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        group = img.shape[0]//3
        alpha = [random.uniform(-0.5, 0.5)*self.var for i in range(3)]
        for grp in range(group):
            idx = grp*3
            for i in range(3):
                img[idx + i] = img[idx + i] + alpha[0]
        return img

class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        group = img.shape[0]//3
        alpha = [1 + random.uniform(-0.5, 0.5)*self.var for i in range(3)]
        for grp in range(group):
            idx = grp*3
            for i in range(3):
                img[idx + i] = img[idx + i]*alpha[0]
        return img

class Gamma(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        group = img.shape[0]//3
        alpha = [1 + random.uniform(-0.5, 0.5)*self.var for i in range(3)]
        for grp in range(group):
            idx = grp*3
            for i in range(3):
                img[idx + i] = img[idx + i]**alpha[0]
        return img

class RandomOrder(object):
    """ 
    Composes several transforms together in random order.
    """

    def __init__(self, transforms, group=2, same_group=True):
        self.transforms = transforms
        self.group = group
        self.same_group = same_group

    def __call__(self, img):
        if self.transforms is None:
            return img
        group = min(self.group, img.shape[0]//3)
        same_group = self.same_group and (group>1)
        range_img = group*3
        if(same_group):
            self.order = torch.randperm(len(self.transforms)) 
            for i in self.order:
                img[:range_img] = self.transforms[i](img[:range_img])
        else:
            for grp in range(group):
                idx = 3*grp
                self.order = torch.randperm(len(self.transforms)) 
                for i in self.order:
                    img[idx:idx+3] = self.transforms[i](img[idx:idx+3])
        img[:range_img] = img[:range_img].clamp(0, 1)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, Jitter=0.4, group=1, same_group=False):
        self.transforms = []
        self.transforms.append(Brightness(Jitter))
        self.transforms.append(Contrast(Jitter))
        self.transforms.append(Saturation(Jitter))
        self.transforms.append(Gamma(Jitter))
        super(ColorJitter, self).__init__(self.transforms, group, same_group)

    def __call__(self, img):
        return super(ColorJitter, self).__call__(img)
        
