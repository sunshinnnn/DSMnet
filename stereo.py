#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models import model_create_by_name as model_create
from losses.loss import losses
import utils.utils as utils


import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

class stereo(object):

    def __init__(self, args):
        self.args = args
        self.use_cuda = torch.cuda.is_available()
        
        # dataloader
        self.dataloader = None
        
        # model
        self.name = args.net
        self.model = model_create(self.name, args.maxdisparity)
        if(self.use_cuda):
            self.model = self.model.cuda()
            #self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda())
        
        # optim
        self.lr = args.lr
        self.alpha = args.alpha
        self.betas = (args.beta1, args.beta2)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
        #self.optim = torch.optim.RMSprop(model.parameters(), lr=self.lr, alpha=self.alpha)
        self.lr_epoch0 = args.lr_epoch0
        self.lr_stride = args.lr_stride
        
        # lossfun
        maxepoch_weight_adjust = 0 if self.args.mode == 'finetune' else args.lr_epoch0*3//4
#        maxepoch_weight_adjust = args.lr_epoch0*3//4
        self.lossfun = losses(loss_name=args.loss_name, count_levels=self.model.count_levels, maxepoch_weight_adjust=maxepoch_weight_adjust)

        # dirpath of saving weight
        self.dirpath = os.path.join(args.output,
                                    "%s_%s" % (args.mode, args.dataset), 
                                    "%s_%s" % (args.net, args.loss_name))
        
        # 
        self.epoch = 0
        self.best_prec = np.inf

        if(os.path.exists(self.args.path_weight)):
            # load pretrained weight
            state_dict = torch.load(self.args.path_weight)['state_dict']
            self.model.load_state_dict(state_dict)
            msg = 'load pretrained weight successly: %s \n' % self.args.path_weight
            logging.info(msg)
        
        if(self.args.mode in ['train', 'finetune']):
            # load checkpoint
            self.load_checkpoint()
        
        msg = "[%s] Model name: %s , loss name: %s , updated epoches: %d \n" % (args.mode, args.net, args.loss_name, self.epoch)
        logging.info(msg)

    def save_checkpoint(self, epoch, best_prec, is_best):
        state = {
                'epoch': epoch,
                'best_prec': best_prec,
                'state_dict': self.model.state_dict(),
                'optim' : self.optim.state_dict(),
                }
        utils.save_checkpoint(state, is_best, dirpath=self.dirpath, filename='model_checkpoint.pkl')
        if(is_best):
            path_save = os.path.join(self.dirpath, 'weight_best.pkl')
            torch.save({'state_dict': self.model.state_dict()}, path_save)
    
    def load_checkpoint(self, best=False):
        state = utils.load_checkpoint(self.dirpath, best)
        if state is not None:
            msg = 'load checkpoint successly: %s \n' % self.dirpath
            logging.info(msg)
            self.epoch = state['epoch'] + 1
            self.best_prec = state['best_prec']
            self.model.load_state_dict(state['state_dict'])
            self.optim.load_state_dict(state['optim'])
    
    def lr_adjust(self, optimizer, epoch0, stride, lr0, epoch):
        if(epoch < epoch0):
            return
        n = ((epoch - epoch0)//stride) + 1
        lr = lr0 * (0.5 ** n)            # 即每stride步，lr = lr /2
        for param_group in optimizer.param_groups:       # 将更新的lr 送入优化器 optimizer 中，进行下一次优化
            param_group['lr'] = lr

    def accuracy(self, dispL, dispL_gt):
        mask_disp = dispL_gt > 0
        disp_diff = (dispL_gt - dispL).abs()
        # epe
        epe = disp_diff[mask_disp].mean()
        # d1
        mask1 = disp_diff[mask_disp] <= 3
        mask2 = (disp_diff[mask_disp] / dispL_gt[mask_disp]) <= 0.05
        pixels_good = (mask1 + mask2) > 0
        d1 = 100 - 100.0 * pixels_good.sum() / mask_disp.sum()
        return d1, epe
    
    def submit(self):
        # 测试信息保存路径
        dirpath_save = os.path.join('submit', "%s_%s" % (self.args.dataset, self.args.flag_model))
        if(not os.path.exists(dirpath_save)):
            os.makedirs(dirpath_save)
        filepath_save = dirpath_save + '.pkl'
        
        # 若已经测试过，直接输出结果。
        filenames, times, D1s, epes = [], [], [], []
        if(os.path.exists(filepath_save)):
            data = torch.load(filepath_save)
            filenames.extend(data['filename'])
            times.extend(data['time'])
            D1s.extend(data['D1'])
            epes.extend(data['epe'])
            for i in range(len(filenames)):
                if(len(D1s) == len(filenames)):
                    print("submit: %s | time: %6.3f, D1: %6.3f, epe: %6.3f" % (filenames[i], times[i], D1s[i], epes[i]))
                    print(np.mean(times), np.mean(D1s), np.mean(epes)) 
                else:
                    print("submit: %s | time: %6.3f" % (filenames[i], times[i]))
                    print(np.mean(times))
            return

        # switch to evaluate mode
        self.model.eval()
    
        # start to predict and save result
        time_end = time.time()
        for batch_idx, (batch, batch_filenames) in enumerate(self.dataloader_val):
            assert batch.shape[2] >= 6
            if(self.use_cuda):
                batch = batch[:, :7].cuda()
            imL = batch[:, :3]
            imR = batch[:, 3:6]
            imL = Variable(imL, volatile=True, requires_grad=False)
            imR = Variable(imR, volatile=True, requires_grad=False)
    
            # compute output
            scale_dispLs, dispLs = self.model(imL, imR)

            # measure elapsed time
            filenames.append(batch_filenames[0])
            times.append(time.time() - time_end)
            time_end = time.time()
            
            # measure accuracy
            if(batch.shape[1] >= 7):
                dispL = batch[:, 6:7]
                d1, epe = self.accuracy(dispLs[0].data, dispL)
                D1s.append(d1)
                epes.append(epe)
                print("submit: %s | time: %6.3f, D1: %6.3f, epe: %6.3f" % (filenames[-1], times[-1], D1s[-1], epes[-1]))
            else:
                print("submit: %s | time: %6.3f" % (filenames[-1], times[-1]))

            # save predict result
            filename_save = filenames[-1].split('.')[0]+'.png'
            path_file = os.path.join(dirpath_save, filename_save)
            cv2.imwrite(path_file, dispLs[0][0, 0].data.cpu().numpy())
        
        # save final result
        data = {
            "filename":filenames, 
            "time":times, 
            "D1":D1s, 
            "epe":epes, 
            }
        torch.save(data, filepath_save)
        if (len(D1s) > 0):
            print(np.mean(times), np.mean(D1s), np.mean(epes)) 
        else:
            print(np.mean(times))


    def start(self):
        args = self.args
        if args.mode == 'test':
            self.validate()
            return
    
        losses, EPEs, D1s, epochs_val, losses_val, EPEs_val, D1s_val = [], [], [], [], [], [], []
        path_val = os.path.join(self.dirpath, "loss.pkl")
        if(os.path.exists(path_val)):
            state_val = torch.load(path_val)
            losses, EPEs, D1s, epochs_val, losses_val, EPEs_val, D1s_val = state_val
        # 开始训练模型
        plt.figure(figsize=(18, 5))
        time_start = time.time()
        epoch0 = self.epoch
        for epoch in range(epoch0, args.epochs):
            self.epoch = epoch
            self.lr_adjust(self.optim, args.lr_epoch0, args.lr_stride, args.lr, epoch) # 自定义的lr_adjust函数，见上
            self.lossfun.Weight_Adjust_levels(epoch)
            msg = 'lr: %.6f | weight of levels: %s' % (self.optim.param_groups[0]['lr'], str(self.lossfun.weight_levels))
            logging.info(msg)
    
            # train for one epoch
            mloss, mEPE, mD1 = self.train()
            losses.append(mloss)
            EPEs.append(mEPE)
            D1s.append(mD1)
    
            if(epoch % self.args.val_freq == 0) or (epoch == args.epochs-1):
                # evaluate on validation set
                mloss_val, mEPE_val, mD1_val = self.validate()
                epochs_val.append(epoch)
                losses_val.append(mloss_val)
                EPEs_val.append(mEPE_val)
                D1s_val.append(mD1_val)
        
                # remember best prec@1 and save checkpoint
                is_best = mD1_val < self.best_prec
                self.best_prec = min(mD1_val, self.best_prec)
                self.save_checkpoint(epoch, self.best_prec, is_best)
                torch.save([losses, EPEs, D1s, epochs_val, losses_val, EPEs_val, D1s_val], path_val)
                
                # plt
                m, n = 1, 3
                ax1 = plt.subplot(m, n, 1)
                ax2 = plt.subplot(m, n, 2)
                ax3 = plt.subplot(m, n, 3)
                plt.sca(ax1); plt.cla(); plt.xlabel("epoch"); plt.ylabel("Loss")
                plt.plot(np.array(losses), label='train'); plt.plot(np.array(epochs_val), np.array(losses_val), label='val'); plt.legend()
                plt.sca(ax2); plt.cla(); plt.xlabel("epoch"); plt.ylabel("EPE")
                plt.plot(np.array(EPEs), label='train'); plt.plot(np.array(epochs_val), np.array(EPEs_val), label='val'); plt.legend()
                plt.sca(ax3); plt.cla(); plt.xlabel("epoch"); plt.ylabel("D1")
                plt.plot(np.array(D1s), label='train'); plt.plot(np.array(epochs_val), np.array(D1s_val), label='val'); plt.legend()
                plt.savefig("check_%s_%s_%s_%s.png" % (args.mode, args.dataset, args.net, args.loss_name))
            
            time_curr = (time.time() - time_start)/3600.0
            time_all =  time_curr*(args.epochs - epoch0)/(epoch + 1 - epoch0)
            msg = 'Progress: %.2f | %.2f (hour)\n' % (time_curr, time_all)
            logging.info(msg)

