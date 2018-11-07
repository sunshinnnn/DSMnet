#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from myDatasets_stereo import dataset_stereo_by_name as dataset_stereo
import myTransforms

def stereoplot_batch(batch, title):
    batch_t = [batch[:, :3], batch[:, 3:6]]
    for idx in range(6, batch.shape[1]):
        batch_t.append(batch[:, idx:idx+1])
    batch = batch_t
    print [tmp.shape for tmp in batch]
    ncol = len(batch)
    assert ncol >= 2
    nrow = min(4, batch[0].shape[0])
    plt.figure(title)
    for i in range(nrow):
        for j in range(ncol):
            image = batch[j][i].cpu().numpy().transpose(1, 2, 0).squeeze()
            #print image.shape
            plt.subplot(nrow, ncol, ncol*i + j + 1); plt.imshow(image) #, plt.cm.gray)
#    plt.pause(1)
    plt.show()

datasets = [] # 需要测试的所有数据集

# stereo-list
root = '/media/qjc/D/data/testimgs/paths_stereo_test'
name_dataset = 'stereo-list'
datasets.append({'root':root, 'name':name_dataset})

# kitti
root = '/media/qjc/D/data/kitti'
name_dataset = 'kitti2015-tr'
datasets.append({'root':root, 'name':name_dataset})
name_dataset = 'kitti2015-te'
datasets.append({'root':root, 'name':name_dataset})

root = '/media/qjc/D/data/kitti'
name_dataset = 'kitti2012-tr'
datasets.append({'root':root, 'name':name_dataset})
name_dataset = 'kitti2012-te'
datasets.append({'root':root, 'name':name_dataset})

root = '/media/qjc/D/data/kitti'
name_dataset = 'kitti-raw'
datasets.append({'root':root, 'name':name_dataset})

root = '/media/qjc/D/data/kitti'
name_dataset = 'kitti2012-tr_kitti2015-tr'
datasets.append({'root':root, 'name':name_dataset})

root = '/media/qjc/D/data/kitti'
name_dataset = 'kitti2012-te_kitti2015-te'
datasets.append({'root':root, 'name':name_dataset})

# sceneflow
root = '/media/qjc/D/data/sceneflow'
name_dataset = 'monkaa'
datasets.append({'root':root, 'name':name_dataset})
name_dataset = 'driving'
datasets.append({'root':root, 'name':name_dataset})
name_dataset = 'flyingthings3d-tr'
datasets.append({'root':root, 'name':name_dataset})
name_dataset = 'flyingthings3d-te'
datasets.append({'root':root, 'name':name_dataset})

# 所有数据集的基本测试
transform = myTransforms.Stereo_train(size_crop=[768, 384], scale_delt=0, shift_max=0) # myTransforms.Stereo_eval() # 
for item in datasets: # [0:-1]:
    try:
        print(item)
        dataset = dataset_stereo(item['name'], item['root'], Train=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=False)
        for batch_idx, (batch, filenames) in enumerate(dataloader):
            if(batch_idx >= 1): break
            print(filenames)
            batch = myTransforms.Stereo_color_batch(batch, myTransforms.Stereo_unnormalize()) # 去正则化
            stereoplot_batch(batch, item['name'])
        print('passed!\n')
    except Exception as err:
        print('An exception happened! \n\t Error message: %s \n' % (err))

# 测试最大平移量
item = datasets[1]
for shift_max in [32, 320]:
    try:
        transform = myTransforms.Stereo_train(size_crop=[768, 384], scale_delt=0, shift_max=shift_max)
        dataset = dataset_stereo(item['name'], item['root'], Train=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=False)
        print('%s [shift_max: %d]' % (item['name'], shift_max))
        for batch_idx, (batch, filenames) in enumerate(dataloader):
            if(batch_idx >= 1): break
            print(filenames)
            batch = myTransforms.Stereo_color_batch(batch, myTransforms.Stereo_unnormalize()) # 去正则化
            stereoplot_batch(batch, item['name'])
        print('passed!\n')
    except Exception as err:
        print('An exception happened! \n\t Error message: %s \n' % (err))

# 测试尺度缩放的变化量
item = datasets[-1]
for scale_delt in [1, 5]:
    try:
        transform = myTransforms.Stereo_train(size_crop=[768, 384], scale_delt=scale_delt, shift_max=0)
        dataset = dataset_stereo(item['name'], item['root'], Train=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=False)
        print('%s [scale_delt: %d]' % (item['name'], scale_delt))
        for batch_idx, (batch, filenames) in enumerate(dataloader):
            if(batch_idx >= 1): break
            print(filenames)
            batch = myTransforms.Stereo_color_batch(batch, myTransforms.Stereo_unnormalize()) # 去正则化
            stereoplot_batch(batch, item['name'])
        print('passed!\n')
    except Exception as err:
        print('An exception happened! \n\t Error message: %s \n' % (err))


filename = filenames[0]
filename = filename.split('.')[0]+'.png'
print(filename)
