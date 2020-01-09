#!/bin/sh
python -W ignore::UserWarning train_semantic_kitti.py --epochs 100 --batch_size 3

python -W ignore::UserWarning train_semantic_kitti.py --epochs 600 --batch_size 3 --pretrained_weights checkpoints/yolov3_ckpt_epoch-8_MAP-0.03.pth