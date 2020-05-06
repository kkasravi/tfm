#!/usr/bin/env bash
cd /tmp/multi-node/models/benchmarks
rm -rf logs checkpoints common/tensorflow/logs && mkdir logs checkpoints
source ~/.bash_profile &&
pyenv activate resnet50v1_5
PYTHON_EXE=$(pyenv which python)
python launch_benchmark.py \
         --verbose \
         --model-name=resnet50v1_5 \
         --docker-image=amr-registry.caas.intel.com/aipg-tf/dev:nightly-cpx-launch-unified-DNNL1-TF-v2-BFLOAT16-HOROVOD-avx512-devel-mkl-py3 \
         --precision=fp32 \
         --mode=training \
         --framework tensorflow \
         --checkpoint=$PWD/checkpoints \
         --output-dir=$PWD/logs \
         --data-location=/tf_dataset_copy/dataset/TF_Imagenet_MiniData_Half \
         --mpi_hostnames='aipg-ra-skx-219.ra.intel.com,aipg-ra-skx-220.ra.intel.com' \
         --mpi_num_processes=4
