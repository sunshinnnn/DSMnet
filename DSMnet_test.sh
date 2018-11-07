#dir_flyingthings3d=/media/qjc/D/data/FlyingThings3D
#root=$dir_flyingthings3d
#dataset=flyingthings3d-tr
dir_kitti=/media/qjc/D/data/kitti
root=$dir_kitti
dataset=kitti2012-tr
net=dispnetcorr # dispnet/dispnetcorr/iresnet/gcnet
loss_name=depthmono-mask # supervised/(common/depthmono/SsSMnet/Cap_ds_lr)[-mask]
mode=train # train/finetune
dataset_train=kitti2015-tr # kitti2015-tr/flyingthings3d-tr
flag_model=${mode}_${dataset_train}/${net}_${loss_name}
path_weight=./output/$flag_model/weight_best.pkl # model_checkpoint.pkl # 
bt=1


python main.py  --mode test --net $net --batchsize $bt --dataset $dataset --root $root \
                   --path_weight $path_weight --flag_model $flag_model
