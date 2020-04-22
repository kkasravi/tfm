#!/usr/bin/env bash
. /etc/profile
STEPS=${STEPS}
BENCHMARKS=${BENCHMARKS}
VERBOSE=${VERBOSE=''}
PWD=${PWD-$(pwd)}
KMP_AFFINITY=${KMP_AFFINITY-'granularity=fine,verbose,compact,1,0'}
MPI_HOSTNAMES=${MPI_HOSTNAMES-''}
NUM_CORES=${NUM_CORES-20}
MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES-''}
MPI_NUM_PROCESSES_PER_SOCKET=${MPI_NUM_PROCESSES_PER_SOCKET-''}
PRECISION=${PRECISION-fp32}
BATCH_SIZE=${BATCH_SIZE-''}
OUTPUT_DIR=${OUTPUT_DIR-''}
PYTHON_EXE=${PYTHON_EXE-''}
export MKL_VERBOSE=1
export MKLDNN_VERBOSE=1
export DNNL_VERBOSE=1
export PYTHONPATH=$BENCHMARKS


rm -rf checkpoints logs && mkdir checkpoints logs && \
    export PYENV_ROOT="$HOME/.pyenv";
    export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH";
    eval "$(pyenv init -)";
    eval "$(pyenv virtualenv-init -)"
pyenv activate resnet50v1_5 &&
python ${BENCHMARKS}/launch_benchmark.py \
         $VERBOSE \
         --model-name=resnet50v1_5 \
         --precision=$PRECISION \
         --mode=training \
         --framework=tensorflow \
         --checkpoint=$PWD/checkpoints \
         $BATCH_SIZE \
         $MPI_HOSTNAMES \
         $MPI_NUM_PROCESSES \
         $MPI_NUM_PROCESSES_PER_SOCKET \
         $OUTPUT_DIR \
         --data-location=/tf_dataset/dataset/TF_Imagenet_FullData --steps=$STEPS 2>&1 | tee $PWD/logs/output.log
