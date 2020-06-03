all: build

build-container:
	dsh add object-detection-2.1.0 gcc-8
	dsh commit kkasravi/object-detection:2.1.0

push-container:
	docker push kkasravi/object-detection:2.1.0

build-venv:
	dsh add rfcn-inference-fp32-venv-2.1.0
	dsh commit kkasravi/rfcn-inference-fp32-venv-build:2.1.0

build-venv-export:
	dsh export /home/appuser/.pyenv kkasravi/rfcn-inference-fp32-venv-build:2.1.0 /home/appuser/.pyenv kkasravi/rfcn-inference-fp32-venv:2.1.0 aipg_labs

push-venv:
	docker push kkasravi/rfcn-inference-fp32-venv:2.1.0

build-model-export:
	dsh export /workspace/rfcn_fp32_inference/models tf-rfcn-fp32-inference:2.1.0 /mnt/ kkasravi/rfcn-inference-fp32-model:2.1.0 aipg_labs
	dsh export /tensorflow/models/research tf-rfcn-fp32-inference:2.1.0 /mnt/research kkasravi/tensorflow-models:2.1.0 aipg_labs

push-model:
	docker push kkasravi/rfcn-inference-fp32-model:2.1.0
	docker push kkasravi/tensorflow-models:2.1.0

build: build-container push-container build-venv build-venv-export push-venv build-model-export push-model
