#!/bin/bash
cd ../mxnet
docker build --pull -t mlperf_trainingv4.0-gigacomputing:unet3d .
