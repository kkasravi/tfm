PWD:=$(shell pwd)
MODELS:=$(shell dsh config get ai-models location)
BENCHMARKS:=$(MODELS)/benchmarks
PYTHON_EXE?=$(shell pyenv which python)
NUM_CORES:=20

all: run1

run:
	python $(BENCHMARKS)/launch_benchmark.py \
			-f tensorflow \
			-r $(PWD) \
			-p int8 \
			-n $(NUM_CORES) \
			-mo inference \
			-m wide_deep_large_ds \
			-b 512 \
			-d $(PWD)/test_preprocessed_test.tfrecords \
			-i 0 \
			-g $(PWD)/wide_deep_int8_pretrained_model.pb \
			--disable-tcmalloc True \
			--benchmark-only \
			--output-dir $(PWD)/logs

test:
	@echo MODELS=$(MODELS) BENCHMARKS=$(BENCHMARKS) PYTHON_EXE=$(PYTHON_EXE)

run1:
	PYTHON_EXE=$(PYTHON_EXE) PYTHONPATH=$(BENCHMARKS) /home/appuser/.pyenv/versions/wide_and_deep_int8_pretrained_model/bin/python $(BENCHMARKS)/common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=recommendation --model-name=wide_deep_large_ds --precision=int8 --mode=inference --benchmark-dir=$(BENCHMARKS) --intelai-models=$(MODELS)/models/recommendation/tensorflow/wide_deep_large_ds --num-cores=20 --batch-size=512 --socket-id=0 --output-dir=$(PWD)/logs --num-processes=1 --num-processes-per-node=1 --num-train-steps=1 --benchmark-only --model-source-dir=$(PWD) --in-graph=$(PWD)/wide_deep_int8_pretrained_model.pb --data-location=$(PWD)/test_preprocessed_test.tfrecords --disable-tcmalloc=True --num_omp_threads=
