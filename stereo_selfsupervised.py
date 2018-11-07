#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
from stereo import stereo
import utils.utils as utils
from utils.utils import *
from torch.autograd import Variable
import myTransforms

import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

class stereo_selfsupervised(stereo):
    
    def __init__(self, args):
        super(stereo_selfsupervised, self).__init__(args)
        self.dataloader_create(args)

    def dataloader_create(self, args):
        from torch.utils.data import DataLoader
        from myDatasets_stereo import dataset_stereo_by_name as dataset_stereo
        args.mode = args.mode.lower()
        if args.mode == 'test' or args.mode == 'submit':
            # dataloader
            transform=myTransforms.Stereo_eval()
            dataset = dataset_stereo(names_dataset=args.dataset, root=args.root, Train=False, transform=transform)
            self.dataloader_val = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4, drop_last=False)
            msg = "Val dataset: %s " % (args.dataset)
            logging.info(msg)
        else:
            # dataloader
            transform=myTransforms.Stereo_Spatial(size_crop=[768, 384], scale_delt=0, shift_max=32)
            dataset = dataset_stereo(names_dataset=args.dataset, root=args.root, Train=True, transform=transform)
            self.dataloader_train = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=False)
            transform=myTransforms.Stereo_ToTensor()
            dataset_val = dataset_stereo(names_dataset=args.dataset_val, root=args.root, Train=False, transform=transform)
            self.dataloader_val = DataLoader(dataset_val, batch_size=args.batchsize, shuffle=False, num_workers=4, drop_last=False)
            msg = "Train dataset: %s , Val dataset: %s " % (args.dataset, args.dataset_val)
            logging.info(msg)

    def flip_lr_tensor(self, tensor):
        data_numpy = np.flip(tensor.cpu().numpy(), axis=-1).copy()
        return torch.from_numpy(data_numpy).type_as(tensor)
        
    def train(self):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        D1 = utils.AverageMeter()
        EPE = utils.AverageMeter()
    
        # switch to train mode
        self.model.train()
    
        time_end = time.time()
        transform = myTransforms.Stereo_color(same_group=True)
        nedge = 64 if self.lossfun.flag_mask else 0
        for i, (batch, filenames) in enumerate(self.dataloader_train):
            assert batch.shape[2] >= 6
            if(self.use_cuda):
                batch = batch.cuda()
            bn, c, h, w = batch.shape
            assert h>2*nedge and w>2*nedge
            batch1 = self.flip_lr_tensor(batch)
            tmp = batch[:, :6, nedge:h-nedge, nedge:w-nedge]
            batch_aug = torch.zeros(tmp.shape).type_as(tmp)
            batch_aug.copy_(tmp) 
            batch_aug = myTransforms.Stereo_color_batch(batch_aug, transform)
            batch1_aug = self.flip_lr_tensor(batch_aug)
            imL_pre = Variable(batch_aug[:, :3], volatile=False, requires_grad=False)
            imR_pre = Variable(batch_aug[:, 3:6], volatile=False, requires_grad=False)
            imL1_pre = Variable(batch1_aug[:, 3:6], volatile=False, requires_grad=False)
            imR1_pre = Variable(batch1_aug[:, :3], volatile=False, requires_grad=False)
            # measure data loading time
            data_time.update(time.time() - time_end)
    
            # compute output
            scale_dispLs, dispLs = self.model(imL_pre, imR_pre)
            scale_dispL1s, dispL1s = self.model(imL1_pre, imR1_pre)

            # compute loss
            imL = Variable(batch[:, :3, nedge:h-nedge, nedge:w-nedge], volatile=False, requires_grad=False)
            imR_src = Variable(batch[:, 3:6], volatile=False, requires_grad=False)
            imL1 = Variable(batch1[:, 3:6, nedge:h-nedge, nedge:w-nedge], volatile=False, requires_grad=False)
            imR1_src = Variable(batch1[:, :3], volatile=False, requires_grad=False)
            argst = {
                    "imR_src": imR_src, "imL": imL, "dispLs": dispLs, 
                    "scale_dispLs": scale_dispLs, "LeftTop": [nedge, nedge], 
                    "imR1_src": imR1_src, "imL1": imL1, "dispL1s": dispL1s, 
                    "scale_dispL1s": scale_dispL1s, "LeftTop1": [nedge, nedge], 
                    }
            loss = self.lossfun(argst)
            losses.update(loss.data[0], imL.size(0))
            
#            if(i < 5):
#                # visualize images
#                import matplotlib.pyplot as plt
#                row, col = 4, 4
#                plt.subplot(row, col, 1); plt.imshow(imL[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 2); plt.imshow(imR_src[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 3); plt.imshow(imL1[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 4); plt.imshow(imR1_src[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 5); plt.imshow(imL_pre[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 6); plt.imshow(imR_pre[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 7); plt.imshow(imL1_pre[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 8); plt.imshow(imR1_pre[0].data.cpu().numpy().transpose(1, 2, 0))
#                for i in range(len(dispLs)):
#                    plt.subplot(row, col, 9+i); plt.imshow(dispLs[i][0, 0].data.cpu().numpy())
#                plt.show()
    
            # compute gradient and do SGD step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # measure accuracy
            if(batch.shape[1] >= 7):
                dispL = batch[:, 6:7, nedge:h-nedge, nedge:w-nedge]
                d1, epe = self.accuracy(dispLs[0].data, dispL)
            else:
                d1, epe = -1, -1
            D1.update(d1, imL.size(0))
            EPE.update(epe, imL.size(0))

            # measure elapsed time
            batch_time.update(time.time() - time_end)
            time_end = time.time()
    
            # 每十步输出一次
            if i % self.args.print_freq == 0:     # default=20
                print('Train: [{0}][{1}/{2}] | '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'D1 {D1.val:.3f} ({D1.avg:.3f}) | '
                      'EPE {EPE.val:.3f} ({EPE.avg:.3f})'.format(
                       self.epoch, i, len(self.dataloader_train), 
                       batch_time=batch_time, data_time=data_time,
                       loss=losses, D1=D1, EPE=EPE))
    
        msg = 'mean train loss: %.3f | mean D1: %.3f | mean EPE: %.3f' % (losses.avg, D1.avg, EPE.avg)
        logging.info(msg)
        return losses.avg, EPE.avg, D1.avg

    def validate(self):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        D1 = utils.AverageMeter()
        EPE = utils.AverageMeter()
    
        # switch to validate mode
        self.model.eval()

        time_end = time.time()
        transform = myTransforms.Stereo_normalize()
        nedge = 0
        for i, (batch, filenames) in enumerate(self.dataloader_val):
            assert batch.shape[2] >= 6
            if(self.use_cuda):
                batch = batch.cuda()
            bn, c, h, w = batch.shape
            assert h>2*nedge and w>2*nedge
            batch1 = self.flip_lr_tensor(batch)
            tmp = batch[:, :6, nedge:h-nedge, nedge:w-nedge]
            batch_aug = torch.zeros(tmp.shape).type_as(tmp)
            batch_aug.copy_(tmp) 
            batch_aug = myTransforms.Stereo_color_batch(batch_aug, transform)
            batch1_aug = self.flip_lr_tensor(batch_aug)
            imL_pre = Variable(batch_aug[:, :3], volatile=True, requires_grad=False)
            imR_pre = Variable(batch_aug[:, 3:6], volatile=True, requires_grad=False)
            imL1_pre = Variable(batch1_aug[:, 3:6], volatile=True, requires_grad=False)
            imR1_pre = Variable(batch1_aug[:, :3], volatile=True, requires_grad=False)
            # measure data loading time
            data_time.update(time.time() - time_end)
    
            # compute output
            scale_dispLs, dispLs = self.model(imL_pre, imR_pre)
            scale_dispL1s, dispL1s = self.model(imL1_pre, imR1_pre)

            # compute loss
            imL = Variable(batch[:, :3, nedge:h-nedge, nedge:w-nedge], volatile=True, requires_grad=False)
            imR_src = Variable(batch[:, 3:6], volatile=True, requires_grad=False)
            imL1 = Variable(batch1[:, 3:6, nedge:h-nedge, nedge:w-nedge], volatile=True, requires_grad=False)
            imR1_src = Variable(batch1[:, :3], volatile=True, requires_grad=False)
            argst = {
                    "imR_src": imR_src, "imL": imL, "dispLs": dispLs, 
                    "scale_dispLs": scale_dispLs, "LeftTop": [nedge, nedge], 
                    "imR1_src": imR1_src, "imL1": imL1, "dispL1s": dispL1s, 
                    "scale_dispL1s": scale_dispL1s, "LeftTop1": [nedge, nedge], 
                    }
            loss = self.lossfun(argst)
            losses.update(loss.data[0], imL.size(0))

#            if(i < 1):
#                # visualize images
#                import matplotlib.pyplot as plt
#                row, col = 4, 4
#                plt.subplot(row, col, 1); plt.imshow(imL[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 2); plt.imshow(imR_src[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 3); plt.imshow(imL1[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 4); plt.imshow(imR1_src[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 5); plt.imshow(imL_pre[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 6); plt.imshow(imR_pre[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 7); plt.imshow(imL1_pre[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 8); plt.imshow(imR1_pre[0].data.cpu().numpy().transpose(1, 2, 0))
#                for i in range(len(dispLs)):
#                    plt.subplot(row, col, 9+i); plt.imshow(dispLs[i][0, 0].data.cpu().numpy())
#                plt.show()

            # measure accuracy
            if(batch.shape[1] >= 7):
                dispL = batch[:, 6:7]
                d1, epe = self.accuracy(dispLs[0].data, dispL)
            else:
                d1, epe = -1, -1
            D1.update(d1, imL.size(0))
            EPE.update(epe, imL.size(0))

            # measure elapsed time
            batch_time.update(time.time() - time_end)
            time_end = time.time()
    
            # 每十步输出一次
            if i % self.args.print_freq == 0:     # default=20
                print('Val: [{0}][{1}/{2}] | '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'D1 {D1.val:.3f} ({D1.avg:.3f}) | '
                      'EPE {EPE.val:.3f} ({EPE.avg:.3f})'.format(
                       self.epoch, i, len(self.dataloader_val), 
                       batch_time=batch_time, data_time=data_time,
                       loss=losses, D1=D1, EPE=EPE))
    
        msg = 'mean test loss: %.3f | mean D1: %.3f | mean EPE: %.3f' % (losses.avg, D1.avg, EPE.avg)
        logging.info(msg)
        return losses.avg, EPE.avg, D1.avg
