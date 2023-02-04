#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
imagenet="data/ImageNet"



select_data="data/CUB/"
imagenet_select_file='./OT_unnorm_cos_imagenet_OT_select_500_classes_train_samples.txt'
num_classes=200
for lr in 0.01; do for wd in 0.00001; do for lambda_u in 1.0; do
python fixmatch_fine_tune_top_1.py \
  -a resnet50 \
  --lr $lr --batch-size 256 --T 1 --mu 1 --threshold 0.00 --lambda-u $lambda_u --wd $wd --num_classes $num_classes --nesterov True --head_back_r 10.0 \
  --pretrained '/moco_v2_800ep_pretrain.pth.tar' \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --imagenet_select $imagenet_select_file \
  --labeled_data_path $select_data \
  --unlabeled_data_path $imagenet --result_file 'OT_select_K_2000/CUB/OT_select_500_classes_BS_256_lr_'$lr'_mu_1_threshold_0.0_lambda_'$lambda_u'_T_1_wd_'$wd'_v2.txt'
done; done; done




