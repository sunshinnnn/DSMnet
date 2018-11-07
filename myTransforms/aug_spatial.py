#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

class Spatial_stereo(object):
    '''空间变换(缩放，平移，随机裁剪)'''
    def __init__(self, size_crop=[768, 384], scale_delt=0.5, shift_max=32):
        self.size_crop = size_crop
        self.ws = 0
        self.hs = 0
        self.scale_rand = 1
        self.scale_delt = scale_delt
        self.shift_max = shift_max

    def shift_stereo(self, img, shift):
        if(shift==0):
            return img
        channel = img.shape[2]
        assert channel >= 6
        # shift
        if(shift > 0):
            img[:, :-shift, 3:6] = img[:, shift:, 3:6]
        else:
            img[:, -shift:, 3:6] = img[:, :shift, 3:6]
        if(channel >= 8):
            if(shift > 0):
                img[:, :-shift, 7:8] = img[:, shift:, 7:8]
            else:
                img[:, -shift:, 7:8] = img[:, :shift, 7:8]
        if(channel > 6):
            for idx in range(6, channel):
                mask = (img[:, :, idx]!=0)
                img[:, :, idx][mask] += shift
        # slice
        if(shift > 0):
            img = img[:, :-shift]
        else:
            img = img[:, -shift:]
        return img
        
    def __call__(self, img):
        assert type(img) is np.ndarray
        assert len(img.shape) == 3
        assert img.shape[2] >= 6
        h0, w0 = img.shape[:2]
        w1, h1 = self.size_crop
        
        if(self.shift_max > 0): # 随机平移
            self.shift_max = min(self.shift_max, w0)
            shift_rand = np.random.randint(0, self.shift_max)
            img = self.shift_stereo(img, shift_rand)
            w0 -= abs(shift_rand)
        
        if(self.scale_delt == 0): # 随机裁剪
            w1 = min(w0, w1)
            h1 = min(h0, h1)
            # 生成随机裁剪位置
            w, h = w1, h1
            ws = np.random.randint(0, w0 - w) if w0>w else 0
            hs = np.random.randint(0, h0 - h) if h0>h else 0
            # 裁剪图像
            img = img[hs:hs+h, ws:ws+w]
        
        else: # 随机裁剪和缩放
            self.scale_rand = 1 + np.random.uniform(0, self.scale_delt)
            if(np.random.rand() > 0.5):
                self.scale_rand = 1.0/self.scale_rand
            # 计算对应的源图像大小
            w = int(w1/self.scale_rand + 0.5)
            h = int(h1/self.scale_rand + 0.5)
            # 超出范围时, 重新调整缩放比例、对应的源图像大小
            scale_adjust = max(float(h)/min(h, h0), float(w)/min(w, w0))
            self.scale_rand *= scale_adjust
            w = int(w/scale_adjust + 0.5)
            h = int(h/scale_adjust + 0.5)
            # 生成随机裁剪位置
            ws = np.random.randint(0, w0 - w) if w0>w else 0
            hs = np.random.randint(0, h0 - h) if h0>h else 0
            # 裁剪图像并重新调整图片大小
            img = img[hs:hs+h, ws:ws+w]
            if(self.scale_rand != 1):
                img = cv2.resize(img, (w1, h1), cv2.INTER_LINEAR)
                if(img.shape[2] > 6):
                    img[:, :, 6:] *= self.scale_rand
        
        return img


class Spatial_edge(object):
    '''空间变换(旋转，缩放，随机裁剪)'''
    def __init__(self, size_crop=[256, 256], scale_delt=1, rotate=True):
        self.size_crop = size_crop
        self.ws = 0
        self.hs = 0
        self.scale_rand = 1
        self.scale_delt = min(4, abs(scale_delt))
        self.rotate = rotate
        self.angle = 0

    def __call__(self, img):
        assert type(img) is np.ndarray
        assert len(img.shape) == 3
        h0, w0 = img.shape[:2]
        w1, h1 = self.size_crop
        
        
        if(self.rotate): # 随机旋转、裁剪和缩放
            self.angle = np.random.uniform(0, 360)
            self.scale_rand = 1.0 + np.random.uniform(0, self.scale_delt)
            if(np.random.rand() > 0.5):
                self.scale_rand = 1.0/self.scale_rand
            # 计算对应的源图像大小
            cos_scale = abs(np.cos(self.angle*np.pi/180) / self.scale_rand)
            sin_scale = abs(np.sin(self.angle*np.pi/180) / self.scale_rand)
            w = int((h1 * sin_scale) + (w1 * cos_scale))
            h = int((h1 * cos_scale) + (w1 * sin_scale))
            # 超出范围时, 重新调整缩放比例、对应的源图像大小
            scale_adjust = max(float(h)/min(h, h0), float(w)/min(w, w0)) 
            self.scale_rand *= scale_adjust
            w = int(w/scale_adjust + 0.5) # 
            h = int(h/scale_adjust + 0.5)
            # 生成随机裁剪位置
            ws = np.random.uniform(-0.5, 0.5) * (w0 - w) if w0>w else 0
            hs = np.random.uniform(-0.5, 0.5) * (h0 - h) if h0>h else 0
            (cX, cY) = ((w0//2) + ws, (h0//2) + hs)
            #print(int(self.scale_rand), int(self.angle), w0, h0, w, h, w1, h1)
            #print(int(cX), int(cY), int(ws), int(hs))
            # 裁剪图像并重新调整图片
            M = cv2.getRotationMatrix2D((cX, cY), self.angle, self.scale_rand)
            M[0, 2] += w1/ 2.0 - cX
            M[1, 2] += h1/ 2.0 - cY
            img = cv2.warpAffine(img, M, (w1, h1))
        
        elif(self.scale_delt == 0): # 随机裁剪
            w1 = min(w0, w1)
            h1 = min(h0, h1)
            # 生成随机裁剪位置
            w, h = w1, h1
            ws = np.random.randint(0, w0 - w) if w0>w else 0
            hs = np.random.randint(0, h0 - h) if h0>h else 0
            # 裁剪图像
            img = img[hs:hs+h, ws:ws+w]
        
        else: # 随机裁剪和缩放
            self.scale_rand = 1 + np.random.uniform(0, self.scale_delt)
            if(np.random.rand() > 0.5):
                self.scale_rand = 1.0/self.scale_rand
            # 计算对应的源图像大小
            w = int(w1/self.scale_rand + 0.5)
            h = int(h1/self.scale_rand + 0.5)
            # 超出范围时, 重新调整缩放比例、对应的源图像大小
            scale_adjust = max(float(h)/min(h, h0), float(w)/min(w, w0))
            self.scale_rand *= scale_adjust
            w = int(w/scale_adjust + 0.5)
            h = int(h/scale_adjust + 0.5)
            # 生成随机裁剪位置
            ws = np.random.randint(0, w0 - w) if w0>w else 0
            hs = np.random.randint(0, h0 - h) if h0>h else 0
            # 裁剪图像并重新调整图片大小
            img = img[hs:hs+h, ws:ws+w]
            if(self.scale_rand != 1):
                img = cv2.resize(img, (w1, h1), cv2.INTER_LINEAR)
                if(img.shape[2] > 6):
                    img[:, :, 6:] *= self.scale_rand

        return img


