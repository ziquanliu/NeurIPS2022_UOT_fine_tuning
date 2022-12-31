import logging
import math
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def get_imagenet_and_CUB(args):
    normalize = transforms.Normalize(mean=imagenet_mean,
                                     std=imagenet_std)
    transform_labeled = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    # 
    #base_labeled_dataset = datasets.ImageFolder(args.labeled_data)
    #base_unlabeled_dataset = datasets.ImageFolder(args.unlabeled_data)
    
    # labeled data are CUB dataset 
    train_labeled_dataset = datasets.ImageFolder(os.path.join(args.labeled_data_path, 'train'),transform=transform_labeled)
    
    
    # unlabeled data are imagenet files, first load the dataset
    train_unlabeled_data_path = os.path.join(args.unlabeled_data_path, 'train')
    train_unlabeled_dataset = datasets.ImageFolder(train_unlabeled_data_path)
    # imagenet_select is the file that contains selected imagenet image names
    if args.imagenet_select != '':
        print('sample data from imagenet......')
        subset_file = open(args.imagenet_select,'r')
        list_imgs = [li.split('\n')[0] for li in subset_file]
        train_unlabeled_dataset.samples = [(
            os.path.join(train_unlabeled_data_path, li.split('_')[0], li),
            train_unlabeled_dataset.class_to_idx[li.split('_')[0]]
            ) for li in list_imgs]
        
    # not sure if this transform will work?????????????????
    # this works!
    train_unlabeled_dataset.transform = TransformFixMatch(mean=imagenet_mean, std=imagenet_std)
    
    test_dataset = datasets.ImageFolder(os.path.join(args.labeled_data_path, 'val'), transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset





class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
            ])
#         self.strong = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(224),
#             RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        return self.normalize(weak)



