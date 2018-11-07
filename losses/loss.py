import sys
sys.path.append("../")
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from SSIM import SSIM
from utils.imwrap import imwrap_BCHW
from utils.utils import imsplot_tensor

import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

flag_test = False
flag_imshow = False

def create_impyramid(im, levels):
    impyramid = [im]
    # pyramid
    for i in range(1, levels):
        impyramid.append(impyramid[-1][:, :, ::2, ::2])
    return impyramid

class loss_stereo(torch.nn.Module):
    def __init__(self):
        super(loss_stereo, self).__init__()
        self.w_ap = 1.0
        self.w_ds = 0.001
        self.w_lr = 0.001
        self.w_m = 0.0001
        self.ssim = SSIM()
    
    def wfun(self, similarity):
        return max(0, similarity - 0.75)/2 + 0.001

    def diff1_dx(self, img):
        assert len(img.shape) == 4
        diff1 = img[:,:,:,1:] - img[:,:,:,:-1]
        return F.pad(diff1, [0,1,0,0])
        
    def diff1_dy(self, img):
        assert len(img.shape) == 4
        diff1 = img[:,:,1:] - img[:,:,:-1]
        return F.pad(diff1, [0,0,0,1])
        
    def diff2_dx(self, img):
        assert len(img.shape) == 4
        diff2 = img[:,:,:,2:] + img[:,:,:,:-2] - img[:,:,:,1:-1] - img[:,:,:,1:-1]
        return F.pad(diff2, [1,1,0,0])

    def diff2_dy(self, img):
        assert len(img.shape) == 4
        diff2 = img[:,:,2:] + img[:,:,:-2] - img[:,:,1:-1] - img[:,:,1:-1]
        return F.pad(diff2, [0,0,1,1])

    def diff_z_dx(self, disp):
        assert len(disp.shape) == 4
        diff_p = (disp[:,:,:,1:-1]/disp[:,:,:,2:]) + (disp[:,:,:,1:-1]/disp[:,:,:,:-2]) - 2
        return F.pad(diff_p, [1,1,0,0])

    def diff_z_dy(self, disp):
        assert len(disp.shape) == 4
        diff_p = (disp[:,:,1:-1]/disp[:,:,2:]) + (disp[:,:,1:-1]/disp[:,:,:-2]) - 2
        return F.pad(diff_p, [0,0,1,1])

    def C_imdiff1(self, img, img_wrap):
        L1_dx = torch.abs(self.diff1_dx(img) - self.diff1_dx(img_wrap))
        L1_dy = torch.abs(self.diff1_dy(img) - self.diff1_dy(img_wrap))
        return L1_dx + L1_dy
    
    def C_ds1(self, img, disp):
        disp_dx = torch.abs(self.diff1_dx(disp))
        disp_dy = torch.abs(self.diff1_dy(disp))

        image_dx = torch.abs(self.diff1_dx(img))
        image_dy = torch.abs(self.diff1_dy(img))
        weights_x = torch.exp(-torch.sum(image_dx, dim=1, keepdim=True))
        weights_y = torch.exp(-torch.sum(image_dy, dim=1, keepdim=True))
    
        #print weights_x.shape, disp_gradients_x.shape
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        return smoothness_x + smoothness_y
    
    def C_ds2(self, img, disp):
        disp_dx = torch.abs(self.diff2_dx(disp))
        disp_dy = torch.abs(self.diff2_dy(disp))

        image_dx = torch.abs(self.diff2_dx(img))
        image_dy = torch.abs(self.diff2_dy(img))
        weights_x = torch.exp(-torch.sum(image_dx, dim=1, keepdim=True))
        weights_y = torch.exp(-torch.sum(image_dy, dim=1, keepdim=True))
    
        #print weights_x.shape, disp_gradients_x.shape
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        return smoothness_x + smoothness_y

    def C_ds3(self, img, disp):
        disp = torch.abs(disp) + 1
        disp_dx = torch.abs(self.diff_z_dx(disp)).clamp(0, 10)
        disp_dy = torch.abs(self.diff_z_dy(disp)).clamp(0, 10)

        image_dx = torch.abs(self.diff1_dx(img))
        image_dy = torch.abs(self.diff1_dy(img))
        mImage_dx = image_dx.mean(-1,True).mean(-2,True).mean(-3,True)
        mImage_dy = image_dy.mean(-1,True).mean(-2,True).mean(-3,True)
        weights_x = torch.exp(-torch.max(image_dx, dim=1, keepdim=True)[0]/(0.5*mImage_dx))
        weights_y = torch.exp(-torch.max(image_dy, dim=1, keepdim=True)[0]/(0.5*mImage_dy))
    
        #print weights_x.shape, disp_gradients_x.shape
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        return smoothness_x + smoothness_y

    def C_ds3t1(self, img, disp):
        disp_dx = torch.abs(self.diff1_dx(disp))
        disp_dy = torch.abs(self.diff1_dy(disp))

        image_dx = torch.abs(self.diff1_dx(img))
        image_dy = torch.abs(self.diff1_dy(img))
        mImage_dx = image_dx.mean(-1,True).mean(-2,True).mean(-3,True)
        mImage_dy = image_dy.mean(-1,True).mean(-2,True).mean(-3,True)
        weights_x = torch.exp(-torch.max(image_dx, dim=1, keepdim=True)[0]/(0.5*mImage_dx))
        weights_y = torch.exp(-torch.max(image_dy, dim=1, keepdim=True)[0]/(0.5*mImage_dy))
    
        #print weights_x.shape, disp_gradients_x.shape
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        return smoothness_x + smoothness_y

    def C_ds3t(self, img, disp):
        disp = torch.abs(disp) + 1
        disp_dx = torch.abs(self.diff_z_dx(disp)).clamp(0, 10)
        disp_dy = torch.abs(self.diff_z_dy(disp)).clamp(0, 10)

        image_dx = torch.abs(self.diff1_dx(img))
        image_dy = torch.abs(self.diff1_dy(img))
        mImage_dx = image_dx.mean(-1,True).mean(-2,True).mean(-3,True)
        mImage_dy = image_dy.mean(-1,True).mean(-2,True).mean(-3,True)
        weights_x = torch.exp(-torch.max(image_dx, dim=1, keepdim=True)[0]/(0.5*mImage_dx))
        weights_y = torch.exp(-torch.max(image_dy, dim=1, keepdim=True)[0]/(0.5*mImage_dy))
    
        #print weights_x.shape, disp_gradients_x.shape
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        return smoothness_x + smoothness_y

class lossfun(loss_stereo):
    def __init__(self, loss_name):
        super(lossfun, self).__init__()
        self.loss_name = loss_name
    
    def loss_common(self, im, im_wrap, disp, disp_wrap, factor=1.0, weight_common=None):
        # ----------------set w_ds and w_lr---------------------
        mask_ap = (im_wrap[:, :1] != 0).detach()
        if(len(mask_ap[mask_ap]) < 1024):
            mask_ap[:] = 1
        img_ssim = self.ssim(im, im_wrap)
        simlary = img_ssim[mask_ap].mean().data[0]
        w = self.wfun(simlary)
        self.w_ds = w
        self.w_lr = w
        
        # ----------------set C_ap and C_lr---------------------
        C_ap = (0.85*0.5)*(1 - img_ssim) + 0.15*(torch.abs(im - im_wrap)) # + self.C_imdiff1(im, im_wrap))
        C_lr = torch.abs(disp - disp_wrap)

        # ---------------------set mask------------------------
        if(weight_common is not None):
            mask_im = ((disp_wrap==0) + mask_ap).detach() > 1
            mask_lr = (disp_wrap==0).detach()
            weight_im = weight_common.clone()
            weight_im[mask_im] = 1.0
            weight_lr = weight_common.clone()
            weight_lr[mask_lr] = 0
            C_ap = C_ap * weight_im
            C_lr = C_lr * weight_lr # mask_lr.float()
            msg = "weight_im maxV: %f, minV: %f ;" % (weight_im.max().data[0], weight_im.min().data[0])
            msg += "weight_lr maxV: %f, minV: %f " % (weight_lr.max().data[0], weight_lr.min().data[0])
            logging.debug(msg)
            

        # ----------------------C_all----------------------------
        C_ap = C_ap.mean()
        C_ds = self.C_ds3(im, disp).mean()
        C_lr = C_lr.mean()
        C = C_ap*self.w_ap + C_ds*(self.w_ds) + C_lr*self.w_lr
        
        # show in screen
        if(flag_test):
            print self.w_ap, self.w_ds, self.w_lr, simlary
            print C.data[0], C_ap.data[0]*self.w_ap, C_ds.data[0]*self.w_ds, C_lr.data[0]*self.w_lr
        return C

    def loss_depthmono(self, im, im_wrap, disp, disp_wrap, factor=1.0, weight_common=None):
        # ----------------set w_ds and w_lr---------------------
        img_ssim = self.ssim(im, im_wrap)
        mask_ap = (im_wrap[:, :1] != 0).detach()
        if(len(mask_ap[mask_ap]) < 1024):
            mask_ap[:] = 1
        simlary = img_ssim[mask_ap].mean().data[0]
        w = self.wfun(simlary)
        self.w_ds = w
        self.w_lr = w
        
        # ----------------set C_ap and C_lr---------------------
        C_ap = (0.85*0.5)*(1 - img_ssim) + 0.15*torch.abs(im - im_wrap)
        C_lr = torch.abs(disp - disp_wrap)

        # ---------------------set mask------------------------
        if(weight_common is not None):
            mask_im = ((disp_wrap==0) + mask_ap).detach() > 1
            mask_lr = (disp_wrap==0)
            weight_im = weight_common.clone()
            weight_im[mask_im] = 1.0
            weight_lr = weight_common.clone()
            weight_lr[mask_lr] = 0
            C_ap = C_ap * weight_im
            C_lr = C_lr * weight_lr
            msg = "weight_im maxV: %f, minV: %f ;" % (weight_im.max().data[0], weight_im.min().data[0])
            msg += "weight_lr maxV: %f, minV: %f " % (weight_lr.max().data[0], weight_lr.min().data[0])
            logging.debug(msg)
            

        # ----------------------C_all----------------------------
        C_ap = C_ap.mean()
        C_ds = self.C_ds1(im, disp).mean()
        C_lr = C_lr.mean()
        C = C_ap*self.w_ap + C_ds*(self.w_ds) + C_lr*self.w_lr
        
        # show in screen
        if(flag_test):
            print self.w_ap, self.w_ds, self.w_lr, simlary
            print C.data[0], C_ap.data[0]*self.w_ap, C_ds.data[0]*self.w_ds, C_lr.data[0]*self.w_lr
        return C

    def loss_Cap_ds_lr(self, im, im_wrap, disp, disp_wrap, factor=1.0, weight_common=None):
        # ----------------set w_ds and w_lr---------------------
        img_ssim = self.ssim(im, im_wrap)
        mask_ap = (im_wrap[:, :1] != 0).detach()
        simlary = img_ssim[mask_ap].mean().data[0]
        w = self.wfun(simlary)
        self.w_ds = w
        self.w_lr = w
        
        # ----------------set C_ap and C_lr---------------------
        C_ap = (0.85*0.5)*(1 - img_ssim) + 0.15*torch.abs(im - im_wrap)
        C_lr = torch.abs(disp - disp_wrap)

        # ---------------------set mask------------------------
        if(weight_common is not None):
            mask_im = ((disp_wrap==0) + mask_ap).detach() > 1
            mask_lr = (disp_wrap==0)
            weight_im = weight_common.clone()
            weight_im[mask_im] = 1.0
            weight_lr = weight_common.clone()
            weight_lr[mask_lr] = 0
            C_ap = C_ap * weight_im
            C_lr = C_lr * weight_lr
            msg = "weight_im maxV: %f, minV: %f ;" % (weight_im.max().data[0], weight_im.min().data[0])
            msg += "weight_lr maxV: %f, minV: %f " % (weight_lr.max().data[0], weight_lr.min().data[0])
            logging.debug(msg)

        # ----------------------C_ap----------------------------
        C_ap = C_ap.mean()
        C = C_ap * self.w_ap

        # ----------------------C_ds----------------------------
        if("ds" in self.loss_name):
            C_ds = self.C_ds1(im, disp).mean()
            C += C_ds * (self.w_ds/factor)

        # ----------------------C_lr----------------------------
        if("lr" in self.loss_name):
            C_lr = C_lr.mean()
            C += C_lr * self.w_lr
        
        # show in screen
        if(flag_test):
            print self.w_ap, self.w_ds, self.w_lr, simlary
            print C.data[0], C_ap.data[0]*self.w_ap, C_ds.data[0]*self.w_ds, C_lr.data[0]*self.w_lr
        return C

    def loss_SsSMnet(self, im, im_wrap, im_wrap1, disp, factor=1.0, weight_common=None):
        # ----------------set w_ds and w_lr---------------------
        img_ssim = self.ssim(im, im_wrap)
        mask_ap = (im_wrap[:, :1] != 0).detach()
        simlary = img_ssim[mask_ap].mean().data[0]
        w = self.wfun(simlary)
        self.w_ds = w
        self.w_lr = w
        
        # ----------------set C_ap and C_lr---------------------
        C_ap = (0.85*0.5)*(1 - img_ssim) + 0.15*(torch.abs(im - im_wrap) + self.C_imdiff1(im, im_wrap))
        C_lr = torch.abs(im - im_wrap1)

        # ---------------------set mask------------------------
        if(weight_common is not None):
            mask_im = ((im_wrap1[:, :1] == 0) + mask_ap).detach() > 1
            mask_lr = (im_wrap1[:, :1] == 0)
            weight_im = weight_common.clone()
            weight_im[mask_im] = 1.0
            weight_lr = weight_common.clone()
            weight_lr[mask_lr] = 0
            C_ap = C_ap * weight_im
            C_lr = C_lr * weight_lr
            msg = "weight_im maxV: %f, minV: %f ;" % (weight_im.max().data[0], weight_im.min().data[0])
            msg += "weight_lr maxV: %f, minV: %f " % (weight_lr.max().data[0], weight_lr.min().data[0])
            logging.debug(msg)
            

        # ----------------------C_all----------------------------
        C_ap = C_ap.mean()
        C_ds = self.C_ds2(im, disp).mean()
        C_lr = C_lr.mean()
        C_mdh = torch.abs(disp).mean()
        C = C_ap*self.w_ap + C_ds*(self.w_ds/factor) + C_lr*self.w_lr + C_mdh*self.w_m
        
        # show in screen
        if(flag_test):
            print self.w_ap, self.w_ds, self.w_lr, simlary
            print C.data[0], C_ap.data[0]*self.w_ap, C_ds.data[0]*self.w_ds, C_lr.data[0]*self.w_lr
        return C

    def loss_supervised(self, disp_gt, disp, flag_smooth=False, factor=1.0):
        mask = disp_gt>0
        if(len(mask[mask])==0):
            return 0
        loss = torch.abs(disp_gt - disp)[mask].mean()
        #loss = loss + 0.001*torch.abs(disp[disp_gt==0]).mean()
        if(flag_smooth):
            disp_dx = self.diff1_dx(disp)
            disp_dy = self.diff1_dy(disp)
            disp_dxdy = (torch.abs(disp_dx) + torch.abs(disp_dy))/factor
            C_smooth = disp_dxdy[mask].clamp(0, 1).mean()
            loss = loss + 0.1*C_smooth
        return loss
    
    
class losses(lossfun):
    def __init__(self, loss_name="supervised", count_levels=1, maxepoch_weight_adjust=1):
        
        # loss_name parse
        self.flag_mask = ("mask" in loss_name)
        loss_name = loss_name.split("-")[0].lower()
        self.loss_names = ["supervised", "depthmono", "SsSMnet".lower(), "Cap_ds_lr".lower(), "common"]
        assert loss_name in self.loss_names or "Cap".lower() in loss_name
        
        #  set lossfun and lossesfun
        super(losses, self).__init__(loss_name)
        self.lossfun = None
        self.lossesfun = None
        self.setlossfun(loss_name)

        # weight_levels
        self.maxepoch_weight_adjust = maxepoch_weight_adjust
        self.count_levels = count_levels
        self.weight_levels = [0]*count_levels
        self.weight_levels[-1] = 1
    
    def setlossfun(self, loss_name):
        if(self.loss_names[0] in loss_name):
            self.lossfun = self.loss_supervised
            self.lossesfun = self.losses_pyramid0
        elif(self.loss_names[1] in loss_name):
            self.lossfun = self.loss_depthmono
            self.lossesfun = self.losses_pyramid1
        elif(self.loss_names[2] in loss_name):
            self.lossfun = self.loss_SsSMnet
            self.lossesfun = self.losses_pyramid2
        elif("Cap".lower() in loss_name):
            self.lossfun = self.loss_Cap_ds_lr
            self.lossesfun = self.losses_pyramid1
        elif(self.loss_names[4] in loss_name):
            self.lossfun = self.loss_common
            self.lossesfun = self.losses_pyramid1

    def Weight_Adjust_levels(self, epoch):
        count_level = self.count_levels
        maxepoch = self.maxepoch_weight_adjust
        self.weight_levels = [0.01]*count_level
        if(count_level == 1 or epoch >= maxepoch):
            self.weight_levels[0] = 1
            return
        x = (1 - epoch/float(maxepoch))*(count_level - 1)
        idx = int(x)
        w = x - idx
        self.weight_levels[idx] = 1 - w
        if(idx < count_level-1):
            self.weight_levels[idx+1] = w       
    
    def weight_common(self, disp, disp_wrap, factor=1.0):
        disp_delt = torch.abs(disp - disp_wrap).detach()/factor
        weight = Variable(torch.zeros(disp_delt.shape), requires_grad=False).type_as(disp_delt)
        mask1 = disp_delt<1
        mask2 = (disp_delt<3) - mask1
        mask3 = disp_delt >= 3
        weight[mask1] = 1.0
        weight[mask2] = 1.0 - (disp_delt[mask2] - 1)*(0.99/2)
        weight[mask3] = 0.01
        msg = "weight maxV: %f, minV: %f" % (weight.max().data[0], weight.min().data[0])
        logging.debug(msg)
        return weight
    
    # losses for loss_supervised
    def losses_pyramid0(self, disp_gt, disps, scale_disps, flag_smooth=False):
        count = len(scale_disps)
        _, _, h, w = disp_gt.shape
        loss = 0
        for i in range(0,  count):
            level = scale_disps[i]
            weight = self.weight_levels[level]
            if(weight <= 0):
                continue
            if(level > 0):
                pred = F.upsample(disps[i], scale_factor=2**level, mode='bilinear')[:, :, :h, :w]
            else:
                pred = disps[i]
            loss = loss + self.lossfun(disp_gt, pred, flag_smooth, factor=1)*weight
        return loss

    # losses for depthmono/common/Cap_ds_lr
    def losses_pyramid1(self, imR_src, imL, dispLs, scale_dispLs, LeftTop, imR1_src, imL1, dispL1s, scale_dispL1s, LeftTop1):
        count = len(scale_dispLs) # count of output
        maxlevel = min(2, max(scale_dispLs))
        for i in range(0, count):
            if(scale_dispLs[i] == maxlevel):
                _, _, h, w = dispLs[maxlevel].shape
        imLs = create_impyramid(imL, maxlevel + 1)
        imL1s = create_impyramid(imL1, maxlevel + 1)
        # compute loss
        loss = 0 
        for i in range(0,  count):
            level = scale_dispLs[i]
            weight = self.weight_levels[level]
            if(weight <= 0):
                continue
            if(level > maxlevel):
                scale_factor = 2**maxlevel
                dispL = F.upsample(dispLs[i], scale_factor=2**(level - maxlevel), mode='bilinear')[:, :, :h, :w]
                dispL1 = F.upsample(dispL1s[i], scale_factor=2**(level - maxlevel), mode='bilinear')[:, :, :h, :w]
            else:
                scale_factor = 2**level
                dispL = dispLs[i]
                dispL1 = dispL1s[i]
            weight_common = None
            weight_common1 = None
            imL_wrap = imwrap_BCHW(imR_src, dispL, fliplr=False, LeftTop=LeftTop, scale_factor=scale_factor)
            imL1_wrap = imwrap_BCHW(imR1_src, dispL1, fliplr=False, LeftTop=LeftTop1, scale_factor=scale_factor)
            dispL_wrap = imwrap_BCHW(dispL1, dispL, fliplr=True, LeftTop=[0, 0], scale_factor=1)
            dispL1_wrap = imwrap_BCHW(dispL, dispL1, fliplr=True, LeftTop=[0, 0], scale_factor=1)
            if(self.flag_mask):
                weight_common = self.weight_common(dispL, dispL_wrap, factor=scale_factor)
                weight_common1 = self.weight_common(dispL1, dispL1_wrap, factor=scale_factor)
            tmp = self.lossfun(imLs[min(level, maxlevel)], imL_wrap, dispL, dispL_wrap, factor=(2**level), weight_common=weight_common)
            tmp1 = self.lossfun(imL1s[min(level, maxlevel)], imL1_wrap, dispL1, dispL1_wrap, factor=(2**level), weight_common=weight_common1)
            loss = loss + (tmp + tmp1)*weight # / (4**level)
            # imshow
            if(flag_imshow and (i==count-1)):
                imsplot_tensor(imL, imL1, imL_wrap, imL1_wrap, 
                               dispLs[0], dispL1s[0], dispL_wrap, dispL1_wrap)
                import matplotlib.pyplot as plt
                plt.savefig("tmp_check.png")
        
        return loss
    
    # losses for SsSMnet
    def losses_pyramid2(self, imR_src, imL, dispLs, scale_dispLs, LeftTop, imR1_src, imL1, dispL1s, scale_dispL1s, LeftTop1):
        count = len(scale_dispLs) # count of output
        maxlevel = min(2, max(scale_dispLs))
        for i in range(0, count):
            if(scale_dispLs[i] == maxlevel):
                _, _, h, w = dispLs[maxlevel].shape
        imLs = create_impyramid(imL, maxlevel + 1)
        imL1s = create_impyramid(imL1, maxlevel + 1)
        # compute loss
        loss = 0 
        for i in range(0,  count):
            level = scale_dispLs[i]
            weight = self.weight_levels[level]
            if(weight == 0):
                continue
            if(level > maxlevel):
                scale_factor = 2**maxlevel
                dispL = F.upsample(dispLs[i], scale_factor=2**(level - maxlevel), mode='bilinear')[:, :, :h, :w]
                dispL1 = F.upsample(dispL1s[i], scale_factor=2**(level - maxlevel), mode='bilinear')[:, :, :h, :w]
            else:
                scale_factor = 2**level
                dispL = dispLs[i]
                dispL1 = dispL1s[i]
            weight_common = None
            weight_common1 = None
            imL_wrap = imwrap_BCHW(imR_src, dispL, fliplr=False, LeftTop=LeftTop, scale_factor=scale_factor)
            imL1_wrap = imwrap_BCHW(imR1_src, dispL1, fliplr=False, LeftTop=LeftTop1, scale_factor=scale_factor)
            imL_wrap1 = imwrap_BCHW(imL1_wrap, dispL, fliplr=True, LeftTop=[0, 0], scale_factor=1)
            imL1_wrap1 = imwrap_BCHW(imL_wrap, dispL1, fliplr=True, LeftTop=[0, 0], scale_factor=1)
            if(self.flag_mask):
                dispL_wrap = imwrap_BCHW(dispL1, dispL, fliplr=True, LeftTop=[0, 0], scale_factor=1)
                dispL1_wrap = imwrap_BCHW(dispL, dispL1, fliplr=True, LeftTop=[0, 0], scale_factor=1)
                weight_common = self.weight_common(dispL, dispL_wrap, factor=scale_factor)
                weight_common1 = self.weight_common(dispL1, dispL1_wrap, factor=scale_factor)
            tmp = self.lossfun(imLs[min(level, maxlevel)], imL_wrap, imL_wrap1, dispL, factor=(2**level), weight_common=weight_common)
            tmp1 = self.lossfun(imL1s[min(level, maxlevel)], imL1_wrap, imL1_wrap1, dispL1, factor=(2**level), weight_common=weight_common1)
            loss = loss + (tmp + tmp1)*weight # / (4**level)
            # imshow
            if(flag_imshow and (i==count-1)):
                imsplot_tensor(imL, imL1, imL_wrap, imL1_wrap, imL_wrap1, imL1_wrap1, dispLs[0], dispL1s[0])
                import matplotlib.pyplot as plt
                plt.savefig("tmp_check.png")
        
        return loss

    
    def forward(self, args):
        return self.lossesfun(**args)

