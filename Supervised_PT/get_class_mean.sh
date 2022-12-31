#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# python main_moco.py \
#   -a resnet50 \
#   --lr 0.03 \
#   --batch-size 256 \
#   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --aug-plus --cos '/home/ziquanliu_ex/CUB_200_2011'

# dataset='SUN'



python get_class_mean.py \
  -a resnet18 \
  --lr 30.0 \
  --gpu 0 \
  --batch-size 300 \
  --num_classes 1000 \
  --pretrained '' \
  --mlp \
  --imagenet_dir '/data-nas2/datasets/ImageNet' \
  --target_data 'ImageNet'
