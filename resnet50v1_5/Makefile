PWD:=$(shell pwd)
MODELS:=$(shell dsh config get data repos ai-models location)
BENCHMARKS:=$(MODELS)/benchmarks
PYTHON_EXE?=$(shell pyenv which python)
NUM_CORES:=20
KMP_AFFINITY:='noverbose,warnings,respect,granularity=core,none'
STEPS?=2

all: inference

env:
	dsh venv 3.7.0 resnet50v1_5

run-help:
	MKL_VERBOSE=1 MKLDNN_VERBOSE=1 DNNL_VERBOSE=1 PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/launch_benchmark.py --help

run:
	NUM_CORES=$(NUM_CORES) KMP_AFFINITY=$(KMP_AFFINITY) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE) ./launch_benchmark.sh

test:
	@echo MODELS=$(MODELS) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE)

inference-help:
	PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --help

inference:
	PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet50v1_5 --precision=int8 --mode=inference --benchmark-dir=$(BENCHMARKS) --intelai-models=$(MODELS)/models/image_recognition/tensorflow/resnet50v1_5 --num-cores=20 --batch-size=128 --socket-id=0 --output-dir=$(PWD)/logs --benchmark-only --model-source-dir=$(PWD) --in-graph=$(PWD)/resnet50v1_5_int8_pretrained_model.pb

training:
	STEPS=$(STEPS) NUM_CORES=$(NUM_CORES) KMP_AFFINITY=$(KMP_AFFINITY) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE) ./launch_training.sh

training1:
	MPI_NUM_PROCESSES_PER_SOCKET=2 MPI_NUM_PROCESSES=4 MKL_VERBOSE=1 MKLDNN_VERBOSE=1 DNNL_VERBOSE=1 PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet50v1_5 --precision=fp32 --mode=training --benchmark-dir=$(BENCHMARKS) --intelai-models=$(MODELS)/models/image_recognition/tensorflow/resnet50v1_5 --num-cores=20 --checkpoint=$(PWD)/checkpoints --data-location=/tf_dataset/dataset/TF_Imagenet_FullData --steps=100
