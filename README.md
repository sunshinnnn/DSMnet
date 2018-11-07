Pytorch implementation of the several Deep Stereo Matching(DSM) algorithm, such as Dispnet, DispnetCorr, GCnet, iResnet, PSMnet. 
The code is only for scientific or personnal use. Please contact me/INRIA for commercial use.
Email: wangyf_1991@163.com

Copyright (C) 2018 yu-feng wang

License:

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>


## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction



## Usage

### Dependencies

- [Python2.7](https://www.python.org/downloads/)
- [PyTorch(0.3.0+)](http://pytorch.org)
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

### Train
As an example, use the following command to train a DSM(such as DispnetCorr) on  a dataset(such as KITTI2015)

```
./train_dispnetcorr_kitti.sh
```

As another example, use the following command to finetune a DSM(such as DispnetCorr) on  a dataset(such as KITTI2015)

```
./finetune_dispnetcorr_kitti.sh
```
You need see the files(train_dispnetcorr_kitti.sh and train_dispnetcorr_kitti.sh) for details. You can alse change the DSM or dataset for train or finetune in the files.

### test
Use the following command to evaluate the trained DSM(such as DispnetCorr) on  a dataset(such as KITTI2015 train data) with ground truth.

```
./test_dispnetcorr_kitti.sh
```
You need see the file(test_dispnetcorr_kitti.sh) for details.

### submit
Use the following command to evaluate the trained DSM(such as DispnetCorr) on  a dataset(such as KITTI2015 test data) without ground truth.

```
./submit_dispnetcorr_kitti.sh
```
You need see the file(submit_dispnetcorr_kitti.sh) for details.

### Pretrained Model
| KITTI |
|---|
|[DispnetC-pretrained-kitti](https://pan.baidu.com/s/1Dy9FitFASBFtTBoKAhodRg)|


## Results

### Qualitative results
#### Left image
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/image_0/000004_10.png">

#### Predicted disparity
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/result_disp_img_0/000004_10.png">

#### Error
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/errors_disp_img_0/000004_10.png">

### Visualization of Receptive Field
We visualize the receptive fields of different settings of PSMNet, full setting and baseline.

Full setting: dilated conv, SPP, stacked hourglass

Baseline: no dilated conv, no SPP, no stacked hourglass

The receptive fields were calculated for the pixel at image center, indicated by the red cross.

<img align="center" src="https://user-images.githubusercontent.com/11732099/37876179-6d6dd97e-307b-11e8-803e-bcdbec29fb94.png">


Any discussions or concerns are welcomed!
