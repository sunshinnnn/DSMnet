#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
from stereo_supervised import stereo_supervised
from stereo_selfsupervised import stereo_selfsupervised

#import traceback
import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of program')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch on stereo matching with Multi-model')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/finetune/test/submit')
    parser.add_argument('--epochs', default=150, type=int, help='epoches of train')
    parser.add_argument('--dataset', default='kitti2015-tr', type=str, help='support option: [kitti2015/kitti2012/flyingthings3d/middlebury]-[tr/te]')
    parser.add_argument('--root', default='./kitti', type=str, help='root path of dataset')
    parser.add_argument('--dataset_val', default='kitti2015-tr', type=str, help='support option: [kitti2015/kitti2012/flyingthings3d/middlebury]-[tr/te]')
    parser.add_argument('--root_val', default='./kitti', type=str, help='root path of dataset_val')
    parser.add_argument('--val_freq', default=1, type=int, help='frequent of Validate model preformance')
    parser.add_argument('--print_freq', default=20, type=int, help='frequent of display current result')
    parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
    parser.add_argument('--loss_name', default="supervised", type=str, help='support option: supervised/(depthmono/SsSMnet/Cap_ds_lr)[-mask]')
    parser.add_argument('--net', default='dispnet', type=str, help='support option: dispnet/dispnetcorr/iresnet/gcnet/...')
    parser.add_argument('--maxdisparity', default=192, type=int, help='')
    parser.add_argument('--path_weight', default="", type=str, help='state dict of model for test or finetune')
    parser.add_argument('--flag_model', default="", type=str, help='flag of model for submit')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for Adam optim')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta1 for Adam optim')
    parser.add_argument('--alpha', default=0.9, type=float, help='alpha for RMSprop optim')
    parser.add_argument('--lr_epoch0', default=50, type=float, help='start number of update to adjust learning rate')
    parser.add_argument('--lr_stride', default=20, type=float, help='stride of update to adjust learning rate')
    parser.add_argument('--output', default='output', type=str, help='dirpath for save model and training losses')
    args = parser.parse_args()
    
    if(args.mode == 'submit'):
        Obj_stereo = stereo_supervised(args)
        Obj_stereo.submit()
    else:
        if('supervised' in args.loss_name):
            print(args.loss_name)
            Obj_stereo = stereo_supervised(args)
            Obj_stereo.start()
        else:
            Obj_stereo = stereo_selfsupervised(args)
            Obj_stereo.start()

