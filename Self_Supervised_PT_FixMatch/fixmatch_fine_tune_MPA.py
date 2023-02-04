#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from dataset_util.ImageNet_CUB import get_imagenet_and_CUB
import pickle

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# FixMatch parameters
parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
parser.add_argument('--head_back_r', default=1.0, type=float,
                    help='lr ratio of head and backbone')
parser.add_argument('--nesterov', default=False, type=bool,
                    help='use nesterov momentum')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='num of classes')
# dataset parameters
parser.add_argument('--pretrained', default='/home/ziquanliu_ex/MoCo/moco_v2_800ep_pretrain.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--labeled_data_path', default='labeled_data', type=str,
                    help='path to labeled data')
parser.add_argument('--result_file', default='resilt_file.txt', type=str,
                    help='path to result txt')
parser.add_argument('--unlabeled_data_path', default='unlabeled_data', type=str,
                    help='path to unlabeled data')
parser.add_argument('--imagenet_select', default='/home/ziquanliu_ex/MoCo/ImageNet_CUB_distance/image_nn_name.txt', type=str,
                    help='path to unlabeled data')
parser.add_argument('--train_iter', default=24, type=int,
                    help='number of training iterations per epoch')

best_acc1 = 0

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    with torch.no_grad():
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)


    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=np.sqrt(2.0/2048.0))
    model.fc.bias.data.zero_()
    print(model.fc.weight.size())
    

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    
    parameter_base = []
    parameter_top = []
    for name,p in model.named_parameters():
#         print(name)
        if name == 'module.fc.weight' or name == 'module.fc.bias':
            parameter_top.append(p)
        else:
            parameter_base.append(p)
    print('top parameters')
    print(len(parameter_top))
    print('backbone parameters')
    print(len(parameter_base))
    

    optimizer = torch.optim.SGD([{'params': parameter_base},
                {'params': parameter_top, 'lr': args.lr*args.head_back_r}], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,nesterov = args.nesterov)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_labeled_dataset, train_unlabeled_dataset, val_dataset = get_imagenet_and_CUB(args)


    if args.distributed:
        train_sampler = DistributedSampler
    else:
        train_sampler = None
    
    
    train_labeled_loader = torch.utils.data.DataLoader(
        train_labeled_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler(train_labeled_dataset),drop_last = True)
    
    train_unlabeled_loader = torch.utils.data.DataLoader(
        train_unlabeled_dataset, batch_size=args.batch_size*args.mu, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler(train_unlabeled_dataset), drop_last = True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    
#     if args.distributed:
#         train_labeled_loader.sampler.set_epoch(0)
#         train_unlabeled_loader.sampler.set_epoch(0)
#     labeled_iter = iter(train_labeled_loader)
#     unlabeled_iter = iter(train_unlabeled_loader)
#     (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
#     if args.gpu is not None:
#         inputs_u_w = inputs_u_w.cuda(args.gpu, non_blocking=True)
#         inputs_u_s = inputs_u_s.cuda(args.gpu, non_blocking=True)
#     inputs_u_w_all = concat_all_gather(inputs_u_w)
#     inputs_u_s_all = concat_all_gather(inputs_u_s)
#     print(inputs_u_w_all.size())
#     print(inputs_u_s_all.size())
    
    
#------------------------------------------------------------------------------------------------- 

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    args.train_iter = len(train_labeled_loader)
    labeled_iter = iter(train_labeled_loader)
    unlabeled_iter = iter(train_unlabeled_loader)
    labeled_epoch = 0
    unlabeled_epoch = 0
    acc1=0.0
    sys.stdout = open(args.result_file[:-4]+".loss.txt", "w")

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, labeled_epoch, args)

        # train for one epoch
        labeled_epoch, unlabeled_epoch = train(train_labeled_loader, train_unlabeled_loader, labeled_iter, unlabeled_iter, model, criterion, optimizer, labeled_epoch, unlabeled_epoch, args)

        # evaluate on validation set
        if epoch%10 == 0:
            acc1 = validate(val_loader, model, criterion, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': labeled_epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
#             if epoch == args.start_epoch:
#                 sanity_check(model.state_dict(), args.pretrained)
    acc1 = validate(val_loader, model, criterion, args)
    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)
    if args.rank % ngpus_per_node ==0:
        result=open(args.result_file,'a')
        result.write('accuracy:'+str(acc1)+', best accuracy: '+str(best_acc1)+'\n')
        result.close()
    sys.stdout.close()


def train(labeled_trainloader, unlabeled_trainloader, labeled_iter, unlabeled_iter, model, criterion, optimizer, labeled_epoch, unlabeled_epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_label = AverageMeter('Loss Label', ':.4e')
    losses_unlabel = AverageMeter('Loss Unlabel', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args.train_iter,
        [batch_time, data_time, losses, losses_label, losses_unlabel, top1, top5],
        prefix="Epoch: [{}]".format(labeled_epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.train()
    end = time.time()
    for batch_i in range(args.train_iter):
        try:
            inputs_x, targets_x = labeled_iter.next()
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_trainloader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_iter.next()

        try:
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
        data_time.update(time.time() - end)
        batch_size = inputs_x.shape[0]
        inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1)
        
#         inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
        targets_x = targets_x.to(args.gpu, non_blocking=True)
        logits = model(inputs)
        logits = de_interleave(logits, 2*args.mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
        max_probs, _ = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
#         print(pseudo_label.size())
#         print(F.log_softmax(logits_u_s).size())
        mask_all = concat_all_gather(mask)
        
        print('select percent: '+str(mask_all.sum().detach().cpu().numpy()/(8.0*batch_size*args.mu)))
        
#         if labeled_epoch >=30 and args.rank % 8 ==0:
#             #mask_targets_u = torch.multiply(mask,targets_u)
#             targets_u_all = concat_all_gather(targets_u)
#             pickle.dump(targets_u_all.detach().cpu().numpy(),open('record_guess_label_fix_class/guess_labels_'+str(batch_i)+'.pkl','wb'))
#             pickle.dump(mask_all.detach().cpu().numpy(),open('record_guess_label_fix_class/mask_labels_'+str(batch_i)+'.pkl','wb'))
        
        
        # naive mean
        Lu = -torch.sum(torch.sum(torch.multiply(F.log_softmax(logits_u_s,dim=-1),pseudo_label),dim=-1, keepdim=False) * mask)/(torch.sum(mask))
        
        # correct mean
#         if torch.sum(mask).item() == 0.0:
#             Lu = torch.tensor(0.0)
#         else:
#             Lu = -torch.sum(torch.sum(torch.multiply(F.log_softmax(logits_u_s,dim=-1),pseudo_label),dim=-1, keepdim=False))/torch.sum(mask)
        

        loss = Lx + args.lambda_u * Lu
        
        acc1, acc5 = accuracy(logits_x, targets_x, topk=(1, 5))
        losses.update(loss.item(), inputs_x.size(0))
        losses_label.update(Lx.item(), inputs_x.size(0))
        losses_unlabel.update(Lu.item(), inputs_x.size(0))
        top1.update(acc1[0], inputs_x.size(0))
        top5.update(acc5[0], inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_i % args.print_freq == 0:
            progress.display(batch_i)
            
    return labeled_epoch, unlabeled_epoch
    


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    confusion_matrix = torch.zeros(args.num_classes, args.num_classes)
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            _, pred = output.topk(1, 1, True, True)
            loss = criterion(output, target)
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
#         print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#               .format(top1=top1, top5=top5))
    
    ngpus_per_node=8
    MPA = 0.0
    if args.rank % ngpus_per_node ==0:
        per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
        print(confusion_matrix.sum())
        MPA = per_class_acc.mean()*100.0
        result=open(args.result_file+'.test_acc.txt','a')
        result.write(str(MPA)+'\n')
        result.close()
    return MPA


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    
    lr_list = [lr,lr*args.head_back_r]
    count=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_list[count]
        count += 1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
