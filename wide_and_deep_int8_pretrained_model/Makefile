PWD:=$(shell pwd)
MODELS:=$(shell dsh config get ai-models location)
BENCHMARKS:=$(MODELS)/benchmarks
PYTHON_EXE?=$(shell pyenv which python)
NUM_CORES:=20
KMP_AFFINITY:='noverbose,warnings,respect,granularity=core,none'

all: run1

run-help:
	MKL_VERBOSE=1 MKLDNN_VERBOSE=1 DNNL_VERBOSE=1 PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/launch_benchmark.py --help

run:
	NUM_CORES=$(NUM_CORES) KMP_AFFINITY=$(KMP_AFFINITY) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE) ./launch_benchmark.sh

test:
	@echo MODELS=$(MODELS) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE)

run1-help:
	PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --help

run1:
	PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=recommendation --model-name=wide_deep_large_ds --precision=int8 --mode=inference --benchmark-dir=$(BENCHMARKS) --intelai-models=$(MODELS)/models/recommendation/tensorflow/wide_deep_large_ds --num-cores=20 --batch-size=512 --socket-id=0 --output-dir=$(PWD)/logs --benchmark-only --model-source-dir=$(PWD) --in-graph=$(PWD)/wide_deep_int8_pretrained_model.pbtxt --data-location=$(PWD)/test_preprocessed_test.tfrecords --disable-tcmalloc=True 
