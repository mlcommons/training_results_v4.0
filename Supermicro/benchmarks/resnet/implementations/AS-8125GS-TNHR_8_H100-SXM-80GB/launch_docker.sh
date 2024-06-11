
path=/raid/mlperf/resnet/dataset
#path=/raid/dldata/imagenet/train-val-recordio-passthrough
#docker run --cap-add=sys_nice --gpus=all --ipc=host -it --rm -v ${PWD}/results:/results -v ${path}:/data temp
docker run --cap-add=sys_nice --gpus=all --ipc=host -it --rm -v ${PWD}/results:/results -v ${path}:/data gitlab-master.nvidia.com/dl/mlperf/optimized:image_classification.20240418
#docker run --cap-add=sys_nice --ipc=host -it --rm -v ${PWD}/results:/results -v /raid/dldata/imagenet/train-val-tfrecord/:/data gitlab-master.nvidia.com/dl/joc/unet3d_pyt:latest
#docker run --cap-add=sys_nice --ipc=host -it --rm -v ${PWD}/results:/results -v /raid/dldata/imagenet/train-val-tfrecord/:/data gitlab-master.nvidia.com/dl/mlperf/optimized:rn50-tf2-reference
