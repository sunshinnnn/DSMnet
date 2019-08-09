# Pytorch implementation of the several Deep Stereo Matching Network(DSMnet)

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Licensing](#Licensing)

## Introduction

Pytorch implementation of the several Deep Stereo Matching Network
### Supported models:

- [Dispnet](http://arxiv.org/abs/1512.02134)
- [DispnetC](http://arxiv.org/abs/1512.02134)
- [GCnet](http://arxiv.org/abs/1703.04309)
- [iResnet](http://arxiv.org/abs/1712.01039v1)
- [PSMnet](http://arxiv.org/pdf/1803.08669)

## Usage

### Dependencies

- [Python2.7](https://www.python.org/downloads/)
- [PyTorch(0.3.0+)](http://pytorch.org)
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

### Train
As an example, use the following command to train a DSM(such as DispnetCorr) on  a dataset(such as KITTI2015)

```
./DSMnet_train_kitti.sh
```

As another example, use the following command to finetune a DSM(such as DispnetCorr) on  a dataset(such as KITTI2015)

```
./DSMnet_finetune.sh
```
You need see the files(train_dispnetcorr_kitti.sh and train_dispnetcorr_kitti.sh) for details. You can alse change the DSM or dataset for train or finetune in the files.


### submit
Use the following command to evaluate the trained DSM(such as DispnetCorr) on  a dataset(such as KITTI2015 test data) without ground truth.

```
./DSMnet_submit.sh
```
You need see the file(submit_dispnetcorr_kitti.sh) for details.

### Pretrained Model
| KITTI | KITTI-raw-ss |
|---|---|
|[DispnetC-pretrained-kitti](https://pan.baidu.com/s/1Dy9FitFASBFtTBoKAhodRg)|[DispnetC-pretrained-kitti-raw-ss](https://pan.baidu.com/s/1AWz9rJVoAuXn2KjCUvllBw)|


## Results

### Qualitative results
#### Left image。
![image](https://github.com/sunshinnnn/DSMnet/blob/master/deploy/10L.png)
<img align="center" src="https://github.com/wyf2017/DSMnet/tree/master/deploy/10L.png">

#### Right image
<img align="center" src="https://github.com/wyf2017/DSMnet/tree/master/deploy/10R.png">

#### Predicted disparity
<img align="center" src="https://github.com/wyf2017/DSMnet/tree/master/deploy/dispL.png">

## Licensing
Unless otherwise stated, the source code and trained Torch and Python
model files are copyright Carnegie Mellon University and licensed
under the [Apache 2.0 License](./LICENSE).
Portions from the following third party sources have
been modified and are included in this repository.
These portions are noted in the source files and are
copyright their respective authors with
the licenses listed.
