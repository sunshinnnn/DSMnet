net=dispnetcorr # dispnet/dispnetcorr/iresnet/gcnet
path_weight=../weight_best_finnal.pkl
path_left=10L.png
path_right=10R.png

python deploy.py --net $net --path_weight $path_weight \
                   --path_left $path_left --path_right $path_right --flip 1
