#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3

imagenet="./datasets/ImageNet"

dataset_name='CUB'
select_data='./dataset/'$dataset_name'/'
imagenet_select_name='./selection/'$dataset_name'/OT_unnorm_cos_imagenet_OT_select_100_classes_train_samples.txt'
num_classes=200
for mu in 1; do for lambda_u in 0.3; do for lr in 0.003; do for wd in 0.0001; do
python fine_tune_top1.py \
  -a resnet18 \
  --lr $lr --batch-size 256 --T 1 --mu $mu --threshold 0.95 --lambda-u $lambda_u --wd $wd --epochs 100 --num_classes $num_classes --nesterov True --head_back_r 10.0 \
  --pretrained '' \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --imagenet_select $imagenet_select_name \
  --labeled_data_path $select_data \
  --unlabeled_data_path $imagenet --result_file './result/'$dataset_name'/BS_256_epoch_100_select_classes_100_nesterov_step_lr_'$lr'_mu_'$mu'_lambda_'$lambda_u'_wd_'$wd'.txt'
done; done; done; done

