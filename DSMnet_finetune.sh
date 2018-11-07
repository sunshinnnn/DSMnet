#dir_flyingthings3d=/media/qjc/D/data/FlyingThings3D
#dataset=flyingthings3d-tr
#root=$dir_flyingthings3d
#dataset_val=flyingthings3d-te
#root_val=$dir_flyingthings3d
dir_kitti=/media/qjc/D/data/kitti
dataset=kitti2015-tr_kitti2012-tr
root=$dir_kitti
dataset_val=kitti2015-tr_kitti2012-tr
root_val=$dir_kitti
net=dispnetcorr # dispnet/dispnetcorr/iresnet/gcnet
loss_name=supervised # supervised/(common/depthmono/SsSMnet/Cap_ds_lr)[-mask]
path_weight=./output/train_kitti2015-tr/dispnetcorr_depthmono-mask/weight_best.pkl
bt=4

python main.py --mode finetune --net $net --loss_name $loss_name --batchsize $bt --epochs 1000 \
               --path_weight $path_weight --lr 0.00002 --lr_epoch0 400 --lr_stride 200 \
               --dataset $dataset --root $root --dataset_val $dataset_val --root_val $root_val \
               --val_freq 20 --print_freq 20

