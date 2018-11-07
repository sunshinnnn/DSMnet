#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from .stereo_check import check_dataset_stereo
from .Dataset_stereo import Dataset_stereo, Datasets_stereo

def dataset_stereo_by_name(names_dataset='kitti2015-tr_kitti2012-tr', root='./kitti', transform=None, Train=True):
    names_dataset = names_dataset.split('_')
    datasets = []
    for name_dataset in names_dataset:
        paths, size_min = check_dataset_stereo(name_dataset, root).getpaths()
        assert len(paths) == 4
        dataset = Dataset_stereo(paths[0], paths[1], paths[2], paths[3], transform=transform, size_min=size_min, Train=Train)
        datasets.append(dataset)
    return Datasets_stereo(datasets)


