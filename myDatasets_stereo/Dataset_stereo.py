#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
加载双目图像及其视差图数据集
'''

import os
import numpy as np
from img_rw import imread, load_disp
from torch.utils.data import Dataset
#import torchvision.transforms as transforms

import logging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

def Datasets_stereo(datasets):
    assert len(datasets)>0
    dataset = datasets[0]
    for i in range(1, len(datasets)):
        assert dataset.Train == datasets[i].Train
        # 合并 paths_img_left, paths_img_right
        dataset.paths_img_left.extend(datasets[i].paths_img_left)
        dataset.paths_img_right.extend(datasets[i].paths_img_right)
        if(dataset.paths_disp_left is not None):
            # 合并 paths_disp_left
            if(datasets[i].paths_disp_left is None):
                dataset.paths_disp_left = None
            else:
                dataset.paths_disp_left.extend(datasets[i].paths_disp_left)
            # 合并 paths_disp_right
            if(dataset.paths_disp_right is not None):
                if(datasets[i].paths_disp_right is None):
                    dataset.paths_disp_right = None
                else:
                    dataset.paths_disp_right.extend(datasets[i].paths_disp_right)
        # 合并最小图片大小
        if(datasets[i].size_min is not None):
            if(dataset.size_min is not None):
                dataset.size_min[0] = min(dataset.size_min[0], datasets[i].size_min[0])
                dataset.size_min[1] = min(dataset.size_min[1], datasets[i].size_min[1])
            else:
                dataset.size_min = datasets[i].size_min
    return dataset

class Dataset_stereo(Dataset):
    def __init__(self, paths_img_left, paths_img_right, paths_disp_left=None, paths_disp_right=None, 
                 transform=None, loader_img=imread, loader_disp=load_disp, size_min=None, Train=False):
        self.paths_img_left = paths_img_left
        self.paths_img_right = paths_img_right
        self.paths_disp_left = paths_disp_left
        self.paths_disp_right = paths_disp_right
        self.loader_img = loader_img
        self.loader_disp = loader_disp
        self.transform = transform
        self.size_min = size_min
        self.Train = Train

    def __len__(self):
        return len(self.paths_img_left)
    
    def Crop_center_bottom(self, img):
        flag_NotCrop = True
        if(self.size_min is not None):
            h_min = self.size_min[0]
            w_min = self.size_min[1]
            flag_NotCrop = False
        if(flag_NotCrop):
            return img
        h, w = img.shape[:2]
        assert (h_min <= h and w_min <= w)
        ws = (w - w_min)//2
        return img[-h_min:, ws:ws+w_min]

    def __getitem__(self, index):
        
        # 加载双目图像及其视差图
        img_left = None
        img_right = None
        disp_left = None
        disp_right = None
        while(True):
            try:
                # 加载左右视角图片
                img_left = self.loader_img(self.paths_img_left[index])
                img_left = self.Crop_center_bottom(img_left)
                img_right = self.loader_img(self.paths_img_right[index])
                img_right = self.Crop_center_bottom(img_right)
                img = np.concatenate([np.float32(img_left), np.float32(img_right)], axis=2)
                # 加载视差图
                if(self.paths_disp_left is not None):
                    disp_left= self.loader_disp(self.paths_disp_left[index])
                    disp_left = self.Crop_center_bottom(disp_left)[:, :, None]
                    img = np.concatenate([img, np.float32(disp_left)], axis=2)
                    if(self.paths_disp_right is not None):
                        disp_right = self.loader_disp(self.paths_disp_right[index])
                        disp_right = self.Crop_center_bottom(disp_right)[:, :, None]
                        img = np.concatenate([img, np.float32(disp_right)], axis=2)
                # 输出debug信息
                msg = ' shape of left image: %s \
                        ' % str(img_left.shape)
                logging.debug(msg)
            except Exception as err:
                # 输出错误信息
                msg = ' A error occurred when loadering img, paths: \n \
                        img_left: %s \n \
                        img_right: %s \n \
                        Error info: %s \
                        ' % (self.paths_img_left[index], self.paths_img_right[index], str(err))
                logging.error(msg)
                
                # 重新选择索引来加载数据
                if(index > 10): index -= np.random.randint(index//2, index)
                else: index += np.random.randint(10, 20)
                index = min(index, len(self.paths_img_left))
            else:
                break

        # 随机翻转
        channel = img.shape[2]
        if(self.Train and channel%2 == 0 and np.random.rand() > 0.5):
            img = np.flip(img, axis=1).copy()
        
        # 数据增强
        if self.transform is not None:
            img = self.transform(img)
        
        # 文件名
        filename = os.path.basename(self.paths_img_left[index])
        return img, filename

