#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
from stereo import stereo
import utils.utils as utils
from utils.utils import *
from torch.autograd import Variable

import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

class stereo_supervised(stereo):
    
    def __init__(self, args):
        super(stereo_supervised, self).__init__(args)
        self.dataloader_create(args)

    def dataloader_create(self, args):
        from torch.utils.data import DataLoader
        from myDatasets_stereo import dataset_stereo_by_name as dataset_stereo
        import myTransforms
        args.mode = args.mode.lower()
        if args.mode == 'test' or args.mode == 'submit':
            # dataloader
            transform=myTransforms.Stereo_eval()
            dataset = dataset_stereo(names_dataset=args.dataset, root=args.root, Train=False, transform=transform)
            self.dataloader_val = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4, drop_last=False)
            msg = "%s dataset: %s , model name: %s " % (args.mode, args.dataset, args.net)
            logging.info(msg)
        else:
            # dataloader
            transform = myTransforms.Stereo_train(size_crop=[768, 384], scale_delt=0, shift_max=32)
            dataset = dataset_stereo(names_dataset=args.dataset, root=args.root, Train=True, transform=transform)
            self.dataloader_train = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=False)
            transform=myTransforms.Stereo_eval()
            dataset_val = dataset_stereo(names_dataset=args.dataset_val, root=args.root, Train=False, transform=transform)
            self.dataloader_val = DataLoader(dataset_val, batch_size=args.batchsize, shuffle=False, num_workers=4, drop_last=False)
            msg = "Train dataset: %s , val dataset: %s " % (args.dataset, args.dataset_val)
            logging.info(msg)
        
    def train(self):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        D1 = utils.AverageMeter()
        EPE = utils.AverageMeter()
    
        # switch to train mode
        self.model.train()
    
        time_end = time.time()
        for i, (batch, filenames) in enumerate(self.dataloader_train):
            # measure data loading time
            assert batch.shape[1] >= 7
            if(self.use_cuda):
                batch = batch[:, :7].cuda()
            imL = batch[:, :3]
            imR = batch[:, 3:6]
            dispL = batch[:, 6:7]
            imL = Variable(imL, volatile=False, requires_grad=False)
            imR = Variable(imR, volatile=False, requires_grad=False)
            dispL = Variable(dispL, volatile=False, requires_grad=False)
            data_time.update(time.time() - time_end)

    
            # compute output
            scale_dispLs, dispLs = self.model(imL, imR)

            # compute loss
            argst = {
                    "disp_gt": dispL, 
                    "disps": dispLs, "scale_disps": scale_dispLs, 
                    "flag_smooth": True, 
                    }
            loss = self.lossfun(argst)
            losses.update(loss.data[0], imL.size(0))
    
#            if(i < 5):
#                # visualize images
#                import matplotlib.pyplot as plt
#                row, col = 4, 3
#                plt.subplot(row, col, 1); plt.imshow(imL[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 2); plt.imshow(imR[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 3); plt.imshow(dispL[0, 0].data.cpu().numpy())
#                for i in range(len(dispLs)):
#                    plt.subplot(row, col, 4+i); plt.imshow(dispLs[i][0, 0].data.cpu().numpy())
#                plt.show()

            # compute gradient and do SGD step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    
            # measure accuracy
            d1, epe = self.accuracy(dispLs[0].data, dispL.data)
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
        losses = utils.AverageMeter()
        D1 = utils.AverageMeter()
        EPE = utils.AverageMeter()

        # switch to evaluate mode
        self.model.eval()
    
        time_end = time.time()
        for i, (batch, filenames) in enumerate(self.dataloader_val):
            assert batch.shape[1] >= 7
            if(self.use_cuda):
                batch = batch[:, :7].cuda()
            imL = batch[:, :3]
            imR = batch[:, 3:6]
            dispL = batch[:, 6:7]
            imL = Variable(imL, volatile=True, requires_grad=False)
            imR = Variable(imR, volatile=True, requires_grad=False)
            dispL = Variable(dispL, volatile=True, requires_grad=False)
    
            # compute output
            scale_dispLs, dispLs = self.model(imL, imR)

            # compute loss
            argst = {
                    "disp_gt": dispL, 
                    "disps": dispLs, "scale_disps": scale_dispLs, 
                    "flag_smooth": True, 
                    }
            loss = self.lossfun(argst)
            losses.update(loss.data[0], imL.size(0))
    
#            if(i < 1):
#                # visualize images
#                import matplotlib.pyplot as plt
#                row, col = 4, 3
#                plt.subplot(row, col, 1); plt.imshow(imL[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 2); plt.imshow(imR[0].data.cpu().numpy().transpose(1, 2, 0)) 
#                plt.subplot(row, col, 3); plt.imshow(dispL[0, 0].data.cpu().numpy())
#                for i in range(len(dispLs)):
#                    plt.subplot(row, col, 4+i); plt.imshow(dispLs[i][0, 0].data.cpu().numpy())
#                plt.show()

            # measure accuracy
            d1, epe = self.accuracy(dispLs[0].data, dispL.data)
            D1.update(d1, imL.size(0))
            EPE.update(epe, imL.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - time_end)
            time_end = time.time()
    
            # 每十步输出一次
            if i % self.args.print_freq == 0:     # default=20
                print('Val: [{0}][{1}/{2}] | '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'D1 {D1.val:.3f} ({D1.avg:.3f}) | '
                      'EPE {EPE.val:.3f} ({EPE.avg:.3f})'.format(
                       self.epoch, i, len(self.dataloader_val), batch_time=batch_time,
                       loss=losses, D1=D1, EPE=EPE))
    
        msg = 'mean test loss: %.3f | mean D1: %.3f | mean EPE: %.3f' % (losses.avg, D1.avg, EPE.avg)
        logging.info(msg)
        return losses.avg, EPE.avg, D1.avg

