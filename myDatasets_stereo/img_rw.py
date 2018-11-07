#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
使用cv2模块读写图片文件
'''

import cv2
import numpy as np
from img_rw_pfm import load_pfm, save_pfm

def load_disp(fname):
    return load_gray(fname)

def load_gray(fname):
    gray = imread(fname)
    if(len(gray.shape)>2):
        gray = gray[:, :, 0]
    gray[gray == np.inf] = 0
    gray[gray == np.nan] = 0
    return gray

def imread(fname):
    if(fname.find('.pfm') > 0):
        return load_pfm(fname)[0]
    else:
        image = cv2.imread(fname)
        image = np.flip(image, axis=2) # bgr --> rgb
        return np.array(image)

def imwrite(fname, image):
    if(fname.find('.pfm') > 0):
        save_pfm(fname, image)
    else:
        image = np.flip(image, axis=2) # rgb --> bgr
        cv2.imwrite(fname, image)
