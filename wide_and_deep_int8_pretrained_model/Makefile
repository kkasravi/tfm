PWD:=$(shell pwd)
NUM_CORES:=20
run:
	python ../../models/benchmarks/launch_benchmark.py \
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


run1:
	cd ../../models/benchmarks && \
	pyenv activate wide_and_deep_int8_pretrained_model && \
	PYTHON_EXE=$(pyenv which python) PYTHONPATH=. /home/appuser/.pyenv/versions/wide_and_deep_int8_pretrained_model/bin/python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=recommendation --model-name=wide_deep_large_ds --precision=int8 --mode=inference --benchmark-dir=/Users/kdkasrav/Github/models/benchmarks --intelai-models=/Users/kdkasrav/Github/models/benchmarks/../models/recommendation/tensorflow/wide_deep_large_ds --num-cores=20 --batch-size=512 --socket-id=0 --output-dir=/Users/kdkasrav/Github/tfm/wide_and_deep_int8_pretrained_model/logs --num-processes=1 --num-processes-per-node=1 --num-train-steps=1 --benchmark-only --model-source-dir=/Users/kdkasrav/Github/tfm/wide_and_deep_int8_pretrained_model --in-graph=/Users/kdkasrav/Github/tfm/wide_and_deep_int8_pretrained_model/wide_deep_int8_pretrained_model.pb --data-location=/Users/kdkasrav/Github/tfm/wide_and_deep_int8_pretrained_model/test_preprocessed_test.tfrecords --disable-tcmalloc=True --num_omp_threads=
