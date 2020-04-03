#!/usr/bin/env bash
. /etc/profile
BENCHMARKS=${BENCHMARKS}
PWD=${PWD-$(pwd)}
KMP_AFFINITY=${KMP_AFFINITY-'granularity=fine,verbose,compact,1,0'}
NUM_CORES=${NUM_CORES-20}
export MKL_VERBOSE=1
export MKLDNN_VERBOSE=1
export DNNL_VERBOSE=1
export PYTHONPATH=$BENCHMARKS

pyenv activate resnet50v1_5
pyenv which python

rm -rf checkpoints logs && mkdir checkpoints logs && \
$PYTHON_EXE ${BENCHMARKS}/launch_benchmark.py \
         --model-name=resnet50v1_5 \
         --precision=fp32 \
         --mode=training \
         --framework=tensorflow \
         --checkpoint=$PWD/checkpoints \
         --data-location=/tf_dataset/dataset/TF_Imagenet_FullData --steps=10 2>&1 | tee $PWD/logs/output.log
