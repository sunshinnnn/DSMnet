#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torchvision.transforms as transforms
from .aug_color import ToTensor_numpy, Normalize, UnNormalize, Lighting, ColorJitter
from .aug_spatial import Spatial_stereo, Spatial_edge

__imagenet_normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
__mnist_normalize = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

def Normalize_Imagenet(group=1):
    prm = __imagenet_normalize.copy()
    prm['group'] = group
    return Normalize(**prm)

def Normalize_mnist():
    return Normalize(**__mnist_normalize)

def UnNormalize_Imagenet(group=1):
    prm = __imagenet_normalize.copy()
    prm['group'] = group
    return UnNormalize(**prm)

def Imagenet_train():
    return transforms.Compose([
                transforms.Scale(256),             # 重新改变大小为size=(w, h) 或 (size, size)
                transforms.RandomSizedCrop(224),   # 随机剪切并resize成给定的size大小
                transforms.RandomHorizontalFlip(),  # 概率为0.5，随机水平翻转。
                transforms.ToTensor(),              # 转化为tensor数据
                ColorJitter(Jitter=0.4, group=1, same_group=False),
                Lighting(alphastd=0.1, group=1, same_group=False),
                Normalize_Imagenet(), 
                ])

def Imagenet_eval():
    return transforms.Compose([
                transforms.Scale(256),             # 重新改变大小为size=(w, h) 或 (size, size)
                transforms.CenterCrop(224),        # 将给定的数据进行中心切割，得到给定的size。
                transforms.ToTensor(),              # 转化为tensor数据
                Normalize_Imagenet(), 
                ])

def Cifar_train():
    return transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 概率为0.5，随机水平翻转。
                transforms.ToTensor(),              # 转化为tensor数据
                ColorJitter(Jitter=0.4, group=1, same_group=False),
                Lighting(alphastd=0.1, group=1, same_group=False),
                Normalize_Imagenet(), 
                ])

def Cifar_eval():
    return transforms.Compose([
                transforms.ToTensor(),              # 转化为tensor数据
                Normalize_Imagenet(), 
                ])

def Mnist_train():
    return transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 概率为0.5，随机水平翻转。
                transforms.ToTensor(),              # 转化为tensor数据
                ColorJitter(Jitter=0.4, group=1, same_group=False),
                Lighting(alphastd=0.1, group=1, same_group=False),
                Normalize_mnist(), 
                ])

def Mnist_eval():
    return transforms.Compose([
                transforms.ToTensor(),              # 转化为tensor数据
                Normalize_mnist(), 
                ])

def edge_train(size_crop=[256, 256], scale_delt=0.4, ratote=False):
    return transforms.Compose([
                Spatial_edge(size_crop, scale_delt, ratote), 
                ToTensor_numpy(channel=4),              # 转化为tensor数据
                ColorJitter(Jitter=0.4, group=1, same_group=True),
                Lighting(alphastd=0.1, group=1, same_group=True),
                Normalize_Imagenet(group=1), 
                ])

def edge_eval():
    return transforms.Compose([
                ToTensor_numpy(channel=4),              # 转化为tensor数据
                Normalize_Imagenet(group=1), 
                ])

def Stereo_train(size_crop=[768, 384], scale_delt=0.4, shift_max=32):
    return transforms.Compose([
                Spatial_stereo(size_crop, scale_delt, shift_max), 
                ToTensor_numpy(channel=6),              # 转化为tensor数据
                #ColorJitter(Jitter=0.4, group=2, same_group=True),
                Lighting(alphastd=0.1, group=2, same_group=True),
                Normalize_Imagenet(group=2), 
                ])

def Stereo_eval():
    return transforms.Compose([
                ToTensor_numpy(channel=6),              # 转化为tensor数据
                Normalize_Imagenet(group=2), 
                ])

def Stereo_Spatial(size_crop=[768, 384], scale_delt=0.4, shift_max=32):
    return transforms.Compose([
                Spatial_stereo(size_crop, scale_delt, shift_max), 
                ToTensor_numpy(channel=6),              # 转化为tensor数据
                ])

def Stereo_color(same_group=True):
    return transforms.Compose([
                ColorJitter(Jitter=0.4, group=2, same_group=same_group),
                Lighting(alphastd=0.1, group=2, same_group=same_group),
                Normalize_Imagenet(group=2), 
                ])

def Stereo_ToTensor():
    return transforms.Compose([
                ToTensor_numpy(channel=6),              # 转化为tensor数据
                ])

def Stereo_normalize():
    return transforms.Compose([
                Normalize_Imagenet(group=2), 
                ])

def Stereo_unnormalize():
    return transforms.Compose([
                UnNormalize_Imagenet(group=2), 
                ])

def Stereo_color_batch(sample_batch, transform):
    assert len(sample_batch.shape)==4
    sample_batch.copy_(sample_batch)
    bn = sample_batch.shape[0]
    for i in range(bn):
        sample_batch[i] = transform(sample_batch[i])
    return sample_batch

