#!/bin/bash
export CUDA_VISIBLE_DEVICES=0



# python get_class_mean.py \
#   -a resnet50 \
#   --lr 30.0 \
#   --gpu 0 \
#   --batch-size 50 \
#   --num_classes 101 \
#   --dataset_name 'caltech' \
#   --pretrained '/home/ziquanliu_ex/MoCo/moco_v2_800ep_pretrain.pth.tar' \
#   --mlp \
#   --imagenet_dir '/home/ziquanliu_ex/caltech'



python get_sample_feature_and_path.py \
  -a resnet50 \
  --lr 30.0 \
  --gpu 0 \
  --batch-size 300 \
  --num_classes 1000 \
  --dataset_name 'ImageNet' \
  --pretrained './moco_v2_800ep_pretrain.pth.tar' \
  --mlp \
  --imagenet_dir './dataset/ImageNet'


