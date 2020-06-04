all: build

build-container:
	dsh restart
	dsh add object-detection-2.1.0
	docker image rmi kkasravi/object-detection:latest || exit 0
	dsh commit kkasravi/object-detection:latest

push-container:
	docker push kkasravi/object-detection:latest

build-venv:
	dsh restart
	dsh add object-detection-2.1.0 rfcn-inference-fp32-venv-2.1.0
	docker image rmi kkasravi/rfcn-inference-fp32-venv-build:latest || exit 0
	dsh commit kkasravi/rfcn-inference-fp32-venv-build:latest

build-venv-export:
	docker image rmi kkasravi/rfcn-inference-fp32-venv:latest || exit 0
	dsh export /home/appuser/.pyenv kkasravi/rfcn-inference-fp32-venv-build:latest /home/appuser/.pyenv kkasravi/rfcn-inference-fp32-venv:latest

push-venv:
	docker push kkasravi/rfcn-inference-fp32-venv:latest

build-model-export:
	docker image rmi kkasravi/rfcn-inference-fp32-model:latest || exit 0
	dsh export /workspace/rfcn_fp32_inference/models tf-rfcn-fp32-inference:2.1.0 /mnt/ kkasravi/rfcn-inference-fp32-model:latest
	docker image rmi kkasravi/tensorflow-models:latest || exit 0
	dsh export /tensorflow/models/research tf-rfcn-fp32-inference:2.1.0 /mnt/research kkasravi/tensorflow-models:latest

push-model:
	docker push kkasravi/rfcn-inference-fp32-model:latest
	docker push kkasravi/tensorflow-models:latest

build: build-container push-container build-venv build-venv-export push-venv build-model-export push-model
