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

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd=0.1, eigval=imagenet_pca["eigval"], eigvec=imagenet_pca["eigvec"]):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var
        self.alpha = None

    def __call__(self, img, same_as_last=False):
        gs = Grayscale()(img)
        if (self.alpha is None) or (not same_as_last):
            self.alpha = random.uniform(0, self.var) # (-self.var, self.var) # 
        return img.lerp(gs, self.alpha) # out = img + alpha * (gs - img)


class Brightness(object):

    def __init__(self, var):
        self.var = var
        self.alpha = None

    def __call__(self, img, same_as_last=False):
        gs = img.new().resize_as_(img).zero_()
        if (self.alpha is None) or (not same_as_last):
            self.alpha = random.uniform(0, self.var) # (-self.var, self.var) # 
        return img.lerp(gs, self.alpha) # out = img + alpha * (gs - img)


class Contrast(object):

    def __init__(self, var):
        self.var = var
        self.alpha = None

    def __call__(self, img, same_as_last=False):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        if (self.alpha is None) or (not same_as_last):
            self.alpha = random.uniform(0, self.var) # (-self.var, self.var) # 
        return img.lerp(gs, self.alpha) # out = img + alpha * (gs - img)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        self.order = None

    def __call__(self, img, same_as_last=False):
        if self.transforms is None:
            return img
        if (self.order is None) or (not same_as_last):
            self.order = torch.randperm(len(self.transforms)) 
        for i in self.order:
            img = self.transforms[i](img, same_as_last)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, same_as_last=False):
        self.transforms = []
        self.same_as_last = same_as_last
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
        super(ColorJitter, self).__init__(self.transforms)

    def __call__(self, img):
        return super(ColorJitter, self).__call__(img, self.same_as_last)
        
