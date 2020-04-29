#!/usr/bin/env bash
rm -rf logs/* checkpoints/*
source ~/.bash_profile &&
PYTHON_EXE=/nfs/pdx/home/kdkasrav/.pyenv/versions/resnet50v1_5/bin/python 
PYTHONPATH=$PYTHONPATH:$HOME/Github/models/models/common/tensorflow/:$HOME/Github/models/models/image_recognition/tensorflow/resnet50v1_5/training/  mpirun -x PYTHONPATH -x LD_LIBRARY_PATH --allow-run-as-root --host 'aipg-ra-skx-219.ra.intel.com:1,aipg-ra-skx-220.ra.intel.com:1' -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -np 2 -bind-to none --map-by slot $PYTHON_EXE $HOME/Github/models/models/image_recognition/tensorflow/resnet50v1_5/training/mlperf_resnet/imagenet_main.py 2 --batch_size=128 --max_train_steps=10 --train_epochs=1 --epochs_between_evals=1 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 22 --version 1 --resnet_size 50 --data_dir=/tf_dataset_copy/dataset/TF_Imagenet_MiniData_Half 2>&1 | tee logs/resnet50v1_5_multinode.log
