#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append("../")
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from myDatasets_stereo.img_rw import imread
from models import model_create_by_name as model_create
import myTransforms

def disp_predict(model, imgL_np, imgR_np, use_cuda=False):
    # convert type of input
    imgL = torch.from_numpy(imgL_np.copy().transpose(2, 0, 1)[None]).float()
    imgR = torch.from_numpy(imgR_np.copy().transpose(2, 0, 1)[None]).float()
    imgL = Variable(imgL, volatile=True, requires_grad=False)
    imgR = Variable(imgR, volatile=True, requires_grad=False)
    if(use_cuda):
        imgL = imgL.cuda()
        imgR = imgR.cuda()
    
    # preprocessing
    transform = myTransforms.Stereo_normalize()
    imgL = transform(imgL/255.0)
    imgR = transform(imgR/255.0)
    
    # predict disparity
    _, disps = model(imgL, imgR, mode="test")
    return disps[0][0, 0].data.cpu().numpy()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch on stereo matching for deploy')
    parser.add_argument('--net', default='dispnetcorr', type=str, help='support option: dispnet/dispnetcorr/iresnet/gcnet/...')
    parser.add_argument('--maxdisparity', default=192, type=int, help='')
    parser.add_argument('--path_weight', default="", type=str, help='path of model state dict to predict disparity')
    parser.add_argument('--path_left', default="10L.png", type=str, help='path of left image')
    parser.add_argument('--path_right', default="10R.png", type=str, help='path of right image')
    parser.add_argument('--flip', default=False, type=bool, help='flag of flip left and right image')
    args = parser.parse_args()
    
    # load img
    imgL = imread(args.path_left)
    imgR = imread(args.path_right)

    # create model and load model weight 
    model = model_create(args.net, args.maxdisparity)
    state_dict = torch.load(args.path_weight)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    # use_cuda
    use_cuda = torch.torch.cuda.is_available()
    if(use_cuda):
        model = model.cuda()
    
    # predict disparity
    if(args.flip):
        imgL1 = np.flip(imgR, axis=1)
        imgR1 = np.flip(imgL, axis=1)
        dispL1 = disp_predict(model, imgL1, imgR1, use_cuda)
        plt.imsave("dispR.png", np.flip(dispL1, axis=-1))
    else:
        dispL = disp_predict(model, imgL, imgR, use_cuda)
        plt.imsave("dispL.png", dispL)


