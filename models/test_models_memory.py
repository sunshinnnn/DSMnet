#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import commands
import time
import torch
import numpy as np
import matplotlib.pyplot as plt 
from torch.autograd import Variable
import sys
sys.path.append("../")
import models

def GPU_used():
    (status, output) = commands.getstatusoutput("nvidia-smi | grep python ")
    #print(output)
    idx = output.rfind("MiB")
    used_MB = int(output[idx-5:idx])
    return used_MB
    
def test(modelname, img_shape):
    assert len(img_shape) == 4
    model = models.model_create_by_name(modelname)
    if(model is None): 
        raise Exception("不支持的模型!**模型名称：{}".format(modelname))
    model = model.cuda().eval()
    imL = Variable(torch.rand(img_shape), volatile=True).cuda()
    imR = Variable(torch.rand(img_shape), volatile=True).cuda()
    model(imL, imR, mode="test")
    time.sleep(1)
    GPU_used_MB = GPU_used()
    print("model: %12s , image shape: %s, GPU memory used(MB): %5d" % (model.name, str(img_shape), GPU_used_MB))
    return model.name, GPU_used_MB


if __name__ == "__main__":
    # [1, 3, 256, 256] # [1, 3, 256, 512] # [1, 3, 256, 1024] 
    # [1, 3, 256, 1242] # [1, 3, 272, 1242] # [1, 3, 375, 1242] 
    # [1, 3, 256, 960] # [1, 3, 352, 960] # [1, 3, 384, 960] # [1, 3, 480, 960] # 
    #img_shape = [1, 3, 256, 512]
    #print("---Overview--- image shape: %s ---" % (str(img_shape)))
    print("支持的模型：")
    for modelname in models.dict_models:
        print(modelname)
    res_test = {}
    for i in [3, ]:
        GPU_used_MBs = []
        name = "err"
        for width in range(512, 2049, 64): # range(512, 1345, 64): # 
            try:
                img_shape = [1, 3, 256, width]
                name, GPU_used_MB = test(models.dict_models[i], img_shape)
                GPU_used_MBs.append([width, GPU_used_MB])
            except Exception as err:
                print(str(err))
        res_test[name] = GPU_used_MBs
    
    for key, val in res_test.items():
        val = np.array(val)
        for tt in val[:, 1]: print(tt)
        plt.plot(val[:, 0], val[:, 1], label=key)
    plt.legend()
    plt.show()

