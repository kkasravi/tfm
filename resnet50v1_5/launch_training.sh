#!/usr/bin/env bash
. /etc/profile
STEPS=${STEPS}
BENCHMARKS=${BENCHMARKS}
PWD=${PWD-$(pwd)}
KMP_AFFINITY=${KMP_AFFINITY-'granularity=fine,verbose,compact,1,0'}
NUM_CORES=${NUM_CORES-20}
MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES-''}
MPI_NUM_PROCESSES_PER_SOCKET=${MPI_NUM_PROCESSES_PER_SOCKET-''}
PRECISION=${PRECISION-fp32}
export MKL_VERBOSE=1
export MKLDNN_VERBOSE=1
export DNNL_VERBOSE=1
export PYTHONPATH=$BENCHMARKS

pyenv activate resnet50v1_5
pyenv which python

rm -rf checkpoints logs && mkdir checkpoints logs && \
$PYTHON_EXE ${BENCHMARKS}/launch_benchmark.py \
         --model-name=resnet50v1_5 \
         --precision=$PRECISION \
         --mode=training \
         --framework=tensorflow \
         --checkpoint=$PWD/checkpoints \
         $MPI_NUM_PROCESSES \
         $MPI_NUM_PROCESSES_PER_SOCKET \
         --data-location=/tf_dataset/dataset/TF_Imagenet_FullData --steps=$STEPS 2>&1 | tee $PWD/logs/output.log
