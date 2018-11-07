#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append("../")
import torch
import torch.nn as nn
from util.imwrap import imwrap_BCHW
from util_conv import net_init, conv2d_bn, deconv2d_bn, Corr1d
from util_fun import myCat2d


flag_bias_t = True
flag_bn = False
activefun_t = nn.ReLU(inplace=True)

class iresnet(nn.Module):
    def __init__(self, maxdisparity=192):
        super(iresnet, self).__init__()
        self.name = "iresnet"
        self.D = maxdisparity
        self.delt = 1e-6
        self.count_levels = 7 # 分辨率层数

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # Stem Block for Multi-scale Shared Features Extraction
        self.conv1 = conv2d_bn(3, 64, kernel_size=7, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.deconv1_s = deconv2d_bn(64, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=2
        self.deconv2_s = deconv2d_bn(128, 32, kernel_size=8, stride=4, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=4
        self.conv_de1_de2 = conv2d_bn(64, 32, kernel_size=1, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        
        # Initial Disparity Estimation Sub-network
        self.corr = Corr1d(kernel_size=1, stride=1, D=81, simfun=None)
        self.redir = conv2d_bn(128, 64, kernel_size=1, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3 = conv2d_bn(81 + 64, 256, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3_1 = conv2d_bn(256, 256, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4 = conv2d_bn(256, 512, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4_1 = conv2d_bn(512, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv5 = conv2d_bn(512, 512, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv5_1 = conv2d_bn(512, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv6 = conv2d_bn(512, 1024, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv6_1 = conv2d_bn(1024, 1024, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr6 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)
        self.deconv5 = deconv2d_bn(1024, 512, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=2
        self.iconv5 = conv2d_bn(1025, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.deconv4 = deconv2d_bn(512, 256, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=2
        self.iconv4 = conv2d_bn(769, 256, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.deconv3 = deconv2d_bn(256, 128, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=2
        self.iconv3 = conv2d_bn(385, 128, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.deconv2 = deconv2d_bn(128, 64, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=2
        self.iconv2 = conv2d_bn(193, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.deconv1 = deconv2d_bn(64, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=2
        self.iconv1 = conv2d_bn(97, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.deconv0 = deconv2d_bn(32, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=2
        self.iconv0 = conv2d_bn(65, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr0 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        # Disparity Refinement Sub-network
        #imwrap up_conv1b2b
        self.r_conv0 = conv2d_bn(65, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.r_conv1 = conv2d_bn(32, 64, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.c_conv1 = conv2d_bn(64, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.r_corr = Corr1d(kernel_size=3, stride=2, D=41, simfun=None)
        self.r_conv1_1 = conv2d_bn(105, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.r_conv2 = conv2d_bn(64, 128, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.r_conv2_1 = conv2d_bn(128, 128, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.r_res2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.r_deconv1 = deconv2d_bn(128, 64, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=2
        self.r_iconv1 = conv2d_bn(129, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.r_res1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.r_deconv0 = deconv2d_bn(64, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t) # stride=2
        self.r_iconv0 = conv2d_bn(65, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.r_res0 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        # init weight
        net_init(self)
        for m in [self.pr6, self.pr5, self.pr4, self.pr3, self.pr2, self.pr1, self.r_res2, self.r_res1, self.r_res0]:
            m.weight.data = m.weight.data*0.1
        
    def forward(self, imL, imR, mode="train", iter = 1):
        assert imL.shape == imR.shape
        maxD = max(self.D, imL.shape[-1]) # 设置最大视差
        out = []
        out_scale = []
        
        # Multi-scale Shared Features Extraction
        conv1L = self.conv1(imL)
        conv1R = self.conv1(imR)
        conv2L = self.conv2(conv1L)
        conv2R = self.conv2(conv1R)
        deconv1L = self.deconv1_s(conv1L)
        deconv1R = self.deconv1_s(conv1R)
        deconv1L = deconv1L[:, :, :imL.shape[-2], :imL.shape[-1]]
        deconv1R = deconv1R[:, :, :imL.shape[-2], :imL.shape[-1]]
        deconv2L = self.deconv2_s(conv2L)
        deconv2R = self.deconv2_s(conv2R)
        deconv1L2L = self.conv_de1_de2(myCat2d(deconv1L, deconv2L))
        deconv1R2R = self.conv_de1_de2(myCat2d(deconv1R, deconv2R))
        
        # Initial Disparity
        corr = self.corr(conv2L, conv2R)
        redir = self.redir(conv2L)
        conv3 = self.conv3(torch.cat([corr, redir], dim=1))
        conv3_1 = self.conv3_1(conv3)
        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)
        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)
        conv6 = self.conv6(conv5_1)
        conv6_1 = self.conv6_1(conv6)
        
        pr6 = self.pr6(conv6_1)
        out.insert(0, pr6)
        out_scale.insert(0, 6)
        pr6 = self.upsample(pr6)
        
        deconv5 = self.deconv5(conv6_1)
        iconv5 = self.iconv5(myCat2d(deconv5, pr6, conv5_1))
        pr5 = self.pr5(iconv5)
        out.insert(0, pr5)
        out_scale.insert(0, 5)
        pr5 = self.upsample(pr5)

        deconv4 = self.deconv4(iconv5)
        iconv4 = self.iconv4(myCat2d(deconv4, pr5, conv4_1))
        pr4 = self.pr4(iconv4)
        out.insert(0, pr4)
        out_scale.insert(0, 4)
        pr4 = self.upsample(pr4)

        deconv3 = self.deconv3(iconv4)
        iconv3 = self.iconv3(myCat2d(deconv3, pr4, conv3_1))
        pr3 = self.pr3(iconv3)
        out.insert(0, pr3)
        out_scale.insert(0, 3)
        pr3 = self.upsample(pr3)

        deconv2 = self.deconv2(iconv3)
        iconv2 = self.iconv2(myCat2d(deconv2, pr3, conv2L))
        pr2 = self.pr2(iconv2)
        out.insert(0, pr2)
        out_scale.insert(0, 2)
        r_pr2 = pr2
        pr2 = self.upsample(pr2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(myCat2d(deconv1, pr2, conv1L))
        pr1 = self.pr1(iconv1)
        out.insert(0, pr1)
        out_scale.insert(0, 1)
        r_pr1 = pr1
        pr1 = self.upsample(pr1)

        deconv0 = self.deconv0(iconv1)
        iconv0 = self.iconv0(myCat2d(deconv0, pr1, deconv1L2L))
        pr0 = self.pr0(iconv0)
        out.insert(0, pr0)
        out_scale.insert(0, 0)
        r_pr0 = pr0
        
        # iterative Disparity Refinement 
        for i in range(iter):
            w_deconv1L2L = imwrap_BCHW(deconv1R2R, -r_pr0)
            reconerror = torch.abs(deconv1L2L - w_deconv1L2L)
            r_conv0 = self.r_conv0(myCat2d(reconerror, r_pr0, deconv1L2L))
            r_conv1 = self.r_conv1(r_conv0)
            c_conv1L = self.c_conv1(conv1L)
            c_conv1R = self.c_conv1(conv1R)
            r_corr = self.r_corr(c_conv1L, c_conv1R)
            r_conv1_1 = self.r_conv1_1(myCat2d(r_conv1, r_corr))
            r_conv2 = self.r_conv2(r_conv1_1)
            r_conv2_1 = self.r_conv2_1(r_conv2)
            
            r_res2 = self.r_res2(r_conv2_1)
            r_pr2 = r_pr2 + r_res2
            out.insert(0, r_pr2)
            out_scale.insert(0, 2)
            
            r_deconv1 = self.r_deconv1(r_conv2_1)
            r_iconv1 = self.r_iconv1(myCat2d(r_deconv1, self.upsample(r_res2), r_conv1_1))
            r_res1 = self.r_res1(r_iconv1)
            r_pr1 = r_pr1 + r_res1
            out.insert(0, r_pr1)
            out_scale.insert(0, 1)
            
            r_deconv0 = self.r_deconv0(r_iconv1)
            r_iconv0 = self.r_iconv0(myCat2d(r_deconv0, self.upsample(r_res1), r_conv0))
            r_res0 = self.r_res0(r_iconv0)
            r_pr0 = r_pr0 + r_res0
            out.insert(0, r_pr0)
            out_scale.insert(0, 0)
        if(mode == "test"): out[-1] = out[-1].clamp(self.delt, maxD)
        
        return out_scale, out

