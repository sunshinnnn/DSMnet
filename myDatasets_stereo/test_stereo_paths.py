#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from stereo_paths import paths_stereo_by_name as paths_stereo

datasets = [] # 需要测试的所有数据集

# stereo-list
root = '/media/qjc/D/data/testimgs/paths_stereo_test'
name_dataset = 'stereo-list'
datasets.append({'root':root, 'name':name_dataset})

root = '/media/qjc/D/data/testimgs/paths_stereo_test-1'
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

# 所有数据集路径的测试
for item in datasets:
    try:
        print(item)
        dataset = paths_stereo(item['name'], rootpath=item['root'])
        paths = dataset.get_paths_idx(dataset.count//2)
        for path in paths:
            print(path) 
        print('passed!\n')

    except Exception as err:
        print('An exception happened! \n\t Error message: %s \n' % (err))

