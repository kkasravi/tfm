all: build

build-container:
	dsh restart
	dsh add image-recognition-2.1.0
	dsh commit kkasravi/image-recognition:latest

push-container:
	docker push kkasravi/image-recognition:latest

build-venv:
	dsh restart
	dsh add image-recognition-2.1.0 resnet50v1_5-training-fp32-venv-2.1.0
	dsh commit kkasravi/resnet50v1_5-training-fp32-venv-build:latest

build-venv-export:
	dsh export /home/appuser/.pyenv kkasravi/resnet50v1_5-training-fp32-venv-build:latest /home/appuser/.pyenv kkasravi/resnet50v1_5-training-fp32-venv:latest

push-venv:
	docker push kkasravi/resnet50v1_5-training-fp32-venv:latest

build-model-export:
	dsh export /workspace/resnet50v1_5_fp32_training tf-resnet50v1-5-fp32-training:2.1.0 /mnt/ kkasravi/resnet50v1_5-training-fp32-model:latest

push-model:
	docker push kkasravi/resnet50v1_5-training-fp32-model:latest

build: build-container push-container build-venv build-venv-export push-venv build-model-export push-model
