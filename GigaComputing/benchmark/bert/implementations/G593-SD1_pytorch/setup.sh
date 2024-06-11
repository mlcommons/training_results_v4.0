#!/bin/bash
cd ../pytorch
docker build --pull -t mlperf_trainingv4.0-gigacomputing:bert .
