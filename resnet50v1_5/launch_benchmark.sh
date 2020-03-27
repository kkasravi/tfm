#!/usr/bin/env bash
BENCHMARKS=${BENCHMARKS}
PWD=${PWD-$(pwd)}
KMP_AFFINITY=${KMP_AFFINITY-'granularity=fine,verbose,compact,1,0'}
NUM_CORES=${NUM_CORES-20}
export MKL_VERBOSE=1
export MKLDNN_VERBOSE=1
export DNNL_VERBOSE=1
export PYTHONPATH=$BENCHMARKS
$PYTHON_EXE ${BENCHMARKS}/launch_benchmark.py \
			-v \
			-f tensorflow \
			-r $PWD \
			-p int8 \
			-n $NUM_CORES \
			-mo inference \
			-m wide_deep_large_ds \
			-b 1 \
			-d $PWD/test_preprocessed_test.tfrecords \
			-i 0 \
			-g $PWD/wide_deep_int8_pretrained_model.pbtxt \
			--benchmark-only \
			--output-dir $PWD/logs
