#dir_flyingthings3d=/media/qjc/D/data/FlyingThings3D
#root=$dir_flyingthings3d
#dataset=flyingthings3d-tr
dir_kitti=/media/qjc/D/data/kitti
root=$dir_kitti
dataset=kitti2015-tr
net=dispnetcorr # dispnet/dispnetcorr/iresnet/gcnet
loss_name=supervised # supervised/(common/depthmono/SsSMnet/Cap_ds_lr)[-mask]
mode=finetune # train/finetune
dataset_train=kitti2015-tr_kitti2012-tr # kitti-raw/kitti2015-tr/flyingthings3d-tr
flag_model=${mode}_${dataset_train}/${net}_${loss_name}
path_weight=./output/$flag_model/weight_best.pkl
bt=1


python main.py  --mode submit --net $net --batchsize $bt --dataset $dataset --root $root \
                   --path_weight $path_weight --flag_model $flag_model
