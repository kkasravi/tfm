#!/usr/bin/env bash
cd ~/Github/models/benchmarks
rm -rf logs checkpoints common/tensorflow/logs && mkdir logs checkpoints
source ~/.bash_profile &&
pyenv activate resnet50v1_5
PYTHON_EXE=$(pyenv which python)
#python launch_benchmark.py \
#         --verbose \
#         --model-name=resnet50v1_5 \
#         --precision=fp32 \
#         --mode=training \
#         --framework tensorflow \
#         --checkpoint=$PWD/checkpoints \
#         --data-location=/tf_dataset_copy/dataset/TF_Imagenet_MiniData_Half \
#         --mpi_hostnames='aipg-ra-skx-219.ra.intel.com,aipg-ra-skx-220.ra.intel.com' \
#         --mpi_num_processes=4 | tee logs/output.log
python launch_benchmark.py --help
