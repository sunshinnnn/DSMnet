#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
check and save paths of stereo dataset
'''

import os, sys
import pickle
from img_rw import imread, load_disp
from stereo_paths import paths_stereo_by_name
import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

dirpath_file = sys.path[0]

class check_dataset_stereo():
    '''check_dataset_stereo'''
    def __init__(self, name='kitti2015-tr', root='./kitti'):
        self.name = name
        self.root = root
        assert os.path.isdir(self.root), ('dataset[%s] do not exist! \n ' % (self.name))
        self.paths_good = None
        self.h = 100000
        self.w = 100000
        self.paths_all = paths_stereo_by_name(self.name, self.root)
        self.name = self.paths_all.flag_dataset
        assert self.paths_all.count>0, ('dataset[%s] do not exist! \n ' % (self.name))
        self.idxs = range(self.paths_all.count)
        logging.debug('dataset count: %d' % len(self.idxs))

    def checkdisp(self, disp):
        assert len(disp.shape) == 2
        # remove extreme case
        th = disp.shape[1]/3
        mask = (disp>th)
        if(mask.mean() > 0.2):
            return False
        return True
        
    def checkgroup(self, paths_group):
        assert type(paths_group) == list
        n = len(paths_group)
        for j in range(n):
            try:
                path = paths_group[j]
                assert os.path.exists(path)
                tmp = None
                if(j < 2):
                    tmp = imread(path)
                else:
                    tmp = load_disp(path)
                    if(not self.checkdisp(tmp)):
                        return False
                assert tmp is not None
                assert len(tmp.shape) >= 2
                if(j == 1):
                    h, w, c = tmp.shape
                    if(self.h > h): self.h = h
                    if(self.w > w): self.w = w
            except Exception as err:
                msg = 'a error occurred when checking img(%s) \nerror info: %s ' % (path, str(err))
                logging.error(msg)
                return False
        return True
    
    def checkpaths_file0(self, savepath, savepath_bad):
        if(os.path.exists(savepath) and os.path.exists(savepath_bad) ):
            f1=file(savepath,'rb')
            paths_good= pickle.load(f1)
            f1.close()
            f1=file(savepath_bad,'rb')
            paths_bad= pickle.load(f1)
            f1.close()
            print('{} has been existed! '.format(savepath))
            print('Basic Info | dataset Name: {}, Count_good: {}, Count_bad: {}'.format(self.name, len(paths_good[0]), len(paths_bad[0])))
            print('Check regain(y/n)?')
            res = raw_input()
            if(res.lower() != u'y'):
                print('passed')
                return [paths_good, paths_bad]
        return None
        
    def checkpaths_file(self, savepath, savepath_bad):
        if(os.path.exists(savepath) and os.path.exists(savepath_bad) ):
            f1=file(savepath,'rb')
            paths_good, self.h, self.w = pickle.load(f1)
            f1.close()
            f1=file(savepath_bad,'rb')
            paths_bad= pickle.load(f1)
            f1.close()
            return [paths_good, paths_bad]
        return None
        
    def checkpaths(self):
        # checking
        logging.info('checking stereo dataset paths: ' + self.name)
        if(len(self.idxs) < 1):
            return None
        paths_good=[]
        paths_bad=[]
        n = len(self.paths_all.get_paths_idx(0))
        for i in range(n):
            paths_good.append([])
            paths_bad.append([])
        count = 0
        count_all = len(self.idxs)
        stride = max(20,  count_all/100);
        sys.stdout.write(' Progress: %d/%d [ %d percent ]' % (count, count_all, 0))
        sys.stdout.flush()
        for idx in self.idxs:
            paths_group = self.paths_all.get_paths_idx(idx)
            if(self.checkgroup(paths_group)):
                for j in range(n):
                    paths_good[j].append(paths_group[j])
            else:
                for j in range(n):
                    paths_bad[j].append(paths_group[j])
            if(count % stride == 0):
                progress = (count*100) // count_all
                sys.stdout.write('\r Progress: %d/%d [ %d percent ]' % (count, count_all, progress))
                sys.stdout.flush()
            count += 1
        print('\r Progress: %d/%d [ %d percent ] Finised!' % (count, count_all, 100))
        return [paths_good, paths_bad]
        
    def checkandsavepath(self):
        if(not os.path.exists(self.root)):
            return
        # create dirpath
        root = self.root
        dirpath = os.path.join(root, 'paths')
        if(not os.path.exists(dirpath)):
            os.mkdir(dirpath)
        savepath = os.path.join(dirpath, '{}.pkl'.format(self.name))
        savepath_bad = savepath.replace('.pkl', '_bad.pkl')

        # check paths from file
        paths = self.checkpaths_file(savepath, savepath_bad)
        if(paths is not None and len(paths)==2 ):
            paths_good, paths_bad = paths
        else:
            # checking
            paths_good, paths_bad = self.checkpaths()
            # save result
            with open(savepath, 'wb') as f:
                pickle.dump([paths_good, self.h, self.w], f, True)
            with open(savepath_bad, 'wb') as f:
                pickle.dump(paths_bad, f, True)
            with open(savepath_bad.replace('.pkl', '.txt'), 'wb') as f:
                f.write('Count_bad: {} \n'.format(len(paths_bad[0])))
                for path in paths_bad[0]:
                    f.write(path + '\n')
        msg = 'dataset Name: {}, Count_good: {}, Count_bad: {}'.format(self.name, len(paths_good[0]), len(paths_bad[0]))
        logging.info(msg)
        self.paths_good = paths_good

    def getpaths(self):
        # check exist file
        if(self.paths_good is None):
            self.checkandsavepath()
        # fill None for the lack of [img_left, img_right, disp_left, disp_right]
        paths_tmp = self.paths_good
        while(len(paths_tmp) < 4):
            paths_tmp.append(None)
        return paths_tmp, [self.h, self.w]

