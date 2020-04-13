PWD:=$(shell pwd)
MODELS:=$(shell dsh config get data repos ai-models location)
BENCHMARKS:=$(MODELS)/benchmarks
PYTHON_EXE?=$(shell pyenv which python)
NUM_CORES:=20
KMP_AFFINITY:='noverbose,warnings,respect,granularity=core,none'
STEPS?=2

all: build

install:
	dsh add autoconf-2.69-11 gcc-9 openmpi-3.1.6 openssh-7.6

venv: install
	dsh venv 3.7.0 resnet50v1_5 && \
		pyenv local resnet50v1_5

build: venv
	pyenv virtualenvs && \
	pip install -r requirements.txt

clean:
	rm -rf checkpoints logs && mkdir checkpoints logs

run-help:
	MKL_VERBOSE=1 MKLDNN_VERBOSE=1 DNNL_VERBOSE=1 PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/launch_benchmark.py --help

run: clean
	NUM_CORES=$(NUM_CORES) KMP_AFFINITY=$(KMP_AFFINITY) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE) ./launch_benchmark.sh

test:
	@echo MODELS=$(MODELS) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE)

inference-help:
	PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --help

inference: clean
	PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet50v1_5 --precision=int8 --mode=inference --benchmark-dir=$(BENCHMARKS) --intelai-models=$(MODELS)/models/image_recognition/tensorflow/resnet50v1_5 --num-cores=20 --batch-size=128 --socket-id=0 --output-dir=$(PWD)/logs --benchmark-only --model-source-dir=$(PWD) --in-graph=$(PWD)/resnet50v1_5_int8_pretrained_model.pb

training: clean
	STEPS=$(STEPS) NUM_CORES=$(NUM_CORES) KMP_AFFINITY=$(KMP_AFFINITY) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE) ./launch_training.sh

training1: clean
	MPI_NUM_PROCESSES_PER_SOCKET=2 MPI_NUM_PROCESSES=4 MKL_VERBOSE=1 MKLDNN_VERBOSE=1 DNNL_VERBOSE=1 PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet50v1_5 --precision=fp32 --mode=training --benchmark-dir=$(BENCHMARKS) --intelai-models=$(MODELS)/models/image_recognition/tensorflow/resnet50v1_5 --num-cores=20 --checkpoint=$(PWD)/checkpoints --data-location=/tf_dataset/dataset/TF_Imagenet_FullData --steps=10

training-mpi: clean
	PRECISION=fp32 MPI_NUM_PROCESSES='--mpi-num-processes=2' MPI_NUM_PROCESSES_PER_SOCKET='--mpi-num-processes-per-socket=1' STEPS=$(STEPS) NUM_CORES=$(NUM_CORES) KMP_AFFINITY=$(KMP_AFFINITY) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE) ./launch_training.sh --output-dir=$(PWD)/logs

training-mpi1: clean
	MPI_NUM_PROCESSES=2 MPI_NUM_PROCESSES_PER_SOCKET=1 MKL_VERBOSE=1 MKLDNN_VERBOSE=1 DNNL_VERBOSE=1 PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) $(PYTHON_EXE) $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet50v1_5 --precision=fp32 --mode=training --benchmark-dir=$(BENCHMARKS) --intelai-models=$(MODELS)/models/image_recognition/tensorflow/resnet50v1_5 --num-cores=20 --checkpoint=$(PWD)/checkpoints --data-location=/tf_dataset/dataset/TF_Imagenet_FullData --steps=10 | tee $(PWD)/logs/output.log

mpirun: clean
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/usr/local/lib/openmpi PYTHONPATH=$$PYTHONPATH:~/Github/models/models/image_recognition/tensorflow/resnet50v1_5/training mpirun --allow-run-as-root -n 2 --map-by socket /home/appuser/.pyenv/versions/resnet50v1_5/bin/python ~/Github/models/models/image_recognition/tensorflow/resnet50v1_5/training/mlperf_resnet/imagenet_main.py 2 --batch_size=64 --max_train_steps=10 --train_epochs=2 --epochs_between_evals=1 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 20 --version 1 --resnet_size 50 --data_dir=/tf_dataset/dataset/TF_Imagenet_FullData --model_dir=$$PWD/checkpoints 2>&1 | tee $$PWD/logs/output.log
