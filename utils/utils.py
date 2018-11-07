#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import shutil    # 文件高级操作
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 将Variable类型数据转化为numpy类型数据
def to_numpy(variable, USE_CUDA):
    '''
    将Variable类型数据转化为numpy类型数据
    '''
    return variable.cpu().data.numpy() if USE_CUDA else variable.data.numpy()

# 将numpy/tensor类型数据转化为Variable类型数据
def to_tensor(npdata, USE_CUDA, volatile=False, requires_grad=False):
    '''
    将numpy/tensor类型数据转化为Variable类型数据
    '''
    tensor = npdata
    if(type(npdata) is np.ndarray):
        tensor = torch.from_numpy(npdata)
    dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    return Variable(tensor, volatile=volatile, requires_grad=requires_grad).type(dtype)

# 根据指定的目录和文件名保存状态数据，若目录不存在则创建
def save_checkpoint(state, is_best, dirpath='./output', filename='model_checkpoint.pkl'):
    '''
    根据指定的目录和文件名保存状态数据，若目录不存在则创建
    '''
    if(not os.path.exists(dirpath)):
        os.makedirs(dirpath)
    path =os.path.join(dirpath, filename)
    patht = path + '.tmp'
    torch.save(state, patht)
    shutil.move(patht, path)
    if is_best:
        shutil.copyfile(path, os.path.join(dirpath, 'model_best.pkl'))

# 根据指定的目录加载状态数据，若文件不存在则返回 None
def load_checkpoint(dirpath='./output', best=False):
    '''
    根据指定的目录加载状态数据，若文件不存在则返回 None
    '''
    filename = 'model_best.pkl' if best else 'model_checkpoint.pkl'
    path = os.path.join(dirpath, filename)
    if(not os.path.exists(path)):
        return None
    return torch.load(path)

# 使用matplotlib.pyplot绘制单个tensor类型图片
def imshow_tensor(img):
    '''
    使用matplotlib.pyplot绘制单个tensor类型图片
    图片尺寸应为(1, c, h, w)
    '''
    if img is None:
        plt.cla()
    else:
        _, c, h, w = img.shape
        if c==3:
            plt.imshow(to_numpy(img[0]).transpose(1, 2, 0))
        else: 
            plt.imshow(to_numpy(img[0, 0]))

# 使用matplotlib.pyplot绘制多个tensor类型图片
def imsplot_tensor(*imgs_tensor):
    """
    使用matplotlib.pyplot绘制多个tensor类型图片
    图片尺寸应为(bn, c, h, w)
    或是单个图片尺寸为(1, c, h, w)的序列
    """
    count = min(8, len(imgs_tensor))
    if(count==0): return
    col = min(2, count)
    row = count//col
    if(count%col > 0):
        row = row + 1
    for i in range(count):
        plt.subplot(row, col, i+1);imshow_tensor(imgs_tensor[i])
    
# 计算并存储参数当前值和平均值
class AverageMeter(object):
    """
    计算并存储参数当前值和平均值
    Computes and stores the average and current value
       batch_time = AverageMeter()
       即 self = batch_time
       则 batch_time 具有__init__，reset，update三个属性，
       直接使用batch_time.update()调用
       功能为：batch_time.update(time.time() - end)
               仅一个参数，则直接保存参数值
        对应定义：def update(self, val, n=1)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        这些有两个参数则求参数val的均值，保存在avg中##不确定##

    """
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cellinted(size, size_cell):
    return size - size%size_cell
    
def mycrop(w0, h0, size_crop, size_cell=64):
    tw = cellinted(w0//4, size_cell)
    th = cellinted(h0, size_cell)
    crop_w = min(tw, size_crop[0])
    crop_h = min(th, size_crop[1])
    sw = np.random.randint(w0//8, w0 + 1 - crop_w)
    sh = np.random.randint(0, h0 + 1 - crop_h)
    ew = sw + crop_w
    eh = sh + crop_h
    return sw, sh, ew, eh
    
def mycrop0(w0, h0, size_crop, size_cell=64):
    tw = cellinted(w0, size_cell)
    th = cellinted(h0, size_cell)
    crop_w = min(tw, size_crop[0])
    crop_h = min(th, size_crop[1])
    sw = np.random.randint(0, w0 + 1 - crop_w)
    sh = np.random.randint(0, h0 + 1 - crop_h)
    ew = sw + crop_w
    eh = sh + crop_h
    return sw, sh, ew, eh
    
def mycrop1(w0, h0, size_crop, size_cell=64):
    tw = cellinted(w0*2//3, size_cell)
    th = cellinted(h0, size_cell)
    crop_w = min(tw, size_crop[0])
    crop_h = min(th, size_crop[1])
    sw = np.random.randint(w0//6, w0 + 1 - crop_w)
    sh = np.random.randint(0, h0 + 1 - crop_h)
    ew = sw + crop_w
    eh = sh + crop_h
    return sw, sh, ew, eh
    
#def impad_tensor(im, min_size=64):
#    bn, c, h, w = im.shape
#    if(h % min_size == 0 and w % min_size == 0):
#        return im
#    pad_h = 0 if h % min_size == 0 else (min_size - h % min_size)
#    pad_w = 0 if w % min_size == 0 else (min_size - w % min_size)
#    return torch.nn.functional.pad(im, (0, pad_w, 0, pad_h))
#

#def create_impyramid(im, levels, flag_avg=True):
#    impyramid = [im]
#    # function
#    if(flag_avg):
#        m = torch.nn.AvgPool2d(2)
#    else:
#        m = torch.nn.MaxPool2d(2)
#    # pyramid
#    for i in range(1, levels):
#        impyramid.append(m(impyramid[-1]))
#
#    return impyramid

#def create_impyramid(im, levels, flag_avg=True):
#    impyramid = [im]
#    # pyramid
#    for i in range(1, levels):
#        impyramid.append(impyramid[-1][:, :, ::2, ::2])
#    return impyramid


## test
#import matplotlib.pyplot as plt
#import numpy as np
#def test_imwrap():
#    im0 = to_tensor(torch.randn(1, 3, 64, 64).numpy())
#    im0[:, :, 17:25, :]=1
#    im0[:, :, :, 13:21]=1
#    im0[:, :, :, -21:-13]=1
#    disps = [to_tensor(torch.ones(1, 1, i, i).numpy()) for i in (4, 8, 16, 32)]
#    im_warps = imwrap_BCHW_pyramid(im0, disps, LeftTop=[13, 17])
#    im_warps1 = imwrap_BCHW_pyramid(im0, disps, fliplr=True, LeftTop=[13, 17])
#    plt.figure(0)
#    plt.imshow(im0.squeeze(0).cpu().data.numpy().transpose(1, 2, 0))
#    plt.figure(1)
#    ax1 = plt.subplot(221);ax2 = plt.subplot(222); 
#    ax3 = plt.subplot(223);ax4 = plt.subplot(224); 
#    plt.sca(ax1); plt.imshow(im_warps[0].squeeze(0).cpu().data.numpy().transpose(1, 2, 0))
#    plt.sca(ax2); plt.imshow(im_warps[1].squeeze(0).cpu().data.numpy().transpose(1, 2, 0))
#    plt.sca(ax3); plt.imshow(im_warps[2].squeeze(0).cpu().data.numpy().transpose(1, 2, 0))
#    plt.sca(ax4); plt.imshow(im_warps[3].squeeze(0).cpu().data.numpy().transpose(1, 2, 0))
#    plt.figure(2)
#    plt.imshow(np.fliplr(im0.squeeze(0).cpu().data.numpy().transpose(1, 2, 0)))
#    plt.figure(3)
#    ax1 = plt.subplot(221);ax2 = plt.subplot(222); 
#    ax3 = plt.subplot(223);ax4 = plt.subplot(224); 
#    plt.sca(ax1); plt.imshow(im_warps1[0].squeeze(0).cpu().data.numpy().transpose(1, 2, 0))
#    plt.sca(ax2); plt.imshow(im_warps1[1].squeeze(0).cpu().data.numpy().transpose(1, 2, 0))
#    plt.sca(ax3); plt.imshow(im_warps1[2].squeeze(0).cpu().data.numpy().transpose(1, 2, 0))
#    plt.sca(ax4); plt.imshow(im_warps1[3].squeeze(0).cpu().data.numpy().transpose(1, 2, 0))
#    plt.show()
#    
#test_imwrap()
#
