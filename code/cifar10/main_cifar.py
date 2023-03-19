import os
import math
import time
import copy
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as T

from resnet import resnet18
from resnet_batchensemble import resnet18_BE
from batchensemble_layers import Ensemble_orderFC
from utils import knn_monitor, fix_seed
# , knn_monitor_BE
from transforms import *
import sys

import getpass
normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])
normalize_c100 = T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
single_transform_c100 = T.Compose([T.ToTensor(), normalize_c100])


class ContrastiveLearningTransform:
    def __init__(self, args, simple=False):
        transforms = [
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]
        transforms_rotation = [
            T.RandomResizedCrop(size=16, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]
        transforms_simple = [
            T.RandomResizedCrop(size=16, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5)
        ]
        if simple:
            self.transform = T.Compose(transforms_simple)
            self.transform_rotation = T.Compose(transforms_simple)
        else:
            self.transform = T.Compose(transforms)
            self.transform_rotation = T.Compose(transforms_rotation)
        if args.pt_dataset == 'cifar10':
            self.single = single_transform
        elif args.pt_dataset == 'cifar100':
            self.single = single_transform_c100

    def __call__(self, x):
        output = [
            self.single(self.transform(x)),
            self.single(self.transform(x)),
            self.single(self.transform_rotation(x))
        ]
        return output

#
# def rotate_images(images):
#     nimages = images.shape[0]
#     n_rot_images = 4 * nimages
#
#     # rotate images all 4 ways at once
#     rotated_images = torch.zeros([n_rot_images, images.shape[1], images.shape[2], images.shape[3]]).cuda()
#     rot_classes = torch.zeros([n_rot_images]).long().cuda()
#
#     rotated_images[:nimages] = images
#     # rotate 90
#     rotated_images[nimages:2 * nimages] = images.flip(3).transpose(2, 3)
#     rot_classes[nimages:2 * nimages] = 1
#     # rotate 180
#     rotated_images[2 * nimages:3 * nimages] = images.flip(3).flip(2)
#     rot_classes[2 * nimages:3 * nimages] = 2
#     # rotate 270
#     rotated_images[3 * nimages:4 * nimages] = images.transpose(2, 3).flip(3)
#     rot_classes[3 * nimages:4 * nimages] = 3
#
#     return rotated_images, rot_classes


def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def negative_cosine_similarity_loss(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


def info_nce_loss(z1, z2, temperature=0.5):
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    logits = z1 @ z2.T
    logits /= temperature
    n = z2.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_dim, bias=False),
                                 nn.BatchNorm1d(out_dim, affine=False))

    def forward(self, x):
        return self.net(x)

class ProjectionMLP_BE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_models=3):
        super().__init__()
        self.net = nn.Sequential(Ensemble_orderFC(in_dim, hidden_dim, num_models, bias=False),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True),
                                 Ensemble_orderFC(hidden_dim, out_dim, num_models, bias=False),
                                 nn.BatchNorm1d(out_dim, affine=False))

    def forward(self, x):
        return self.net(x)

class PredictionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class MultiProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_models = 1):
        super().__init__()
        self.projectors = nn.ModuleDict()
        for i in range(num_models):
            self.projectors.update({'member'+str(i):  nn.Sequential(
                                            nn.Linear(in_dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, out_dim),
                                        )})
    def forward(self, x_list):
        outs = [self.projectors['member'+str(i)](x1) for i,x1 in enumerate(x_list)]
        return outs


class Branch(nn.Module):
    def __init__(self, args, encoder=None):
        super().__init__()
        dim_proj = [int(x) for x in args.dim_proj.split(',')]
        if not args.BE:
            if encoder:
                self.encoder = encoder
            else:
                self.encoder = resnet18()
            self.projector = ProjectionMLP(512, dim_proj[0], dim_proj[1])

        else:
            self.encoder = resnet18_BE(num_models=args.num_be)
            if args.sep_proj:
                self.projector = MultiProjectionMLP(512, dim_proj[0], dim_proj[1], num_models=args.num_be)
            else:
                self.projector = ProjectionMLP_BE(512, dim_proj[0], dim_proj[1], num_models=args.num_be)

        if args.loss == 'simclr':
            self.predictor2 = nn.Sequential(nn.Linear(512, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2048, args.num_rotate_classes))  # output layer
        else:
            self.predictor2 = nn.Sequential(nn.Linear(512, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.Linear(2048, args.num_rotate_classes))  # output layer

        self.sep_proj = args.sep_proj
        self.num_models = args.num_be
    def forward(self, x):
        out = self.encoder(x)
        if not self.sep_proj:
            out = self.projector(out)
        else:
            out = out.chunk(self.num_models)
            out = self.projector(out) # returns list
            out = torch.cat(out, dim=0)
        return out


def knn_loop(encoder, train_loader, test_loader, BE=False, num_models=3):
    if not BE:
        accuracy = knn_monitor(net=encoder.cuda(),
                               memory_data_loader=train_loader,
                               test_data_loader=test_loader,
                               device='cuda',
                               k=200,
                               hide_progress=True)
    else:
        accuracy = knn_monitor_BE(net=encoder.cuda(),
                               memory_data_loader=train_loader,
                               test_data_loader=test_loader,
                               device='cuda',
                               k=200,
                               hide_progress=True,
                               num_models=num_models)
    return accuracy


def ssl_loop(args, encoder=None):
    if args.checkpoint_path:
        print('checkpoint provided => moving to evaluation')
        main_branch = Branch(args, encoder=encoder).cuda()
        saved_dict = torch.load(os.path.join(args.checkpoint_path))['state_dict']
        main_branch.load_state_dict(saved_dict)
        file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log'), 'a')
        file_to_update.write(f'evaluating {args.checkpoint_path}\n')
        return main_branch.encoder, file_to_update

    # logging
    file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log'), 'w')


    # dataset
    if args.pt_dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                args.data_root, train=True, transform=ContrastiveLearningTransform(args,simple=args.basic_aug), download=True
            ),
            shuffle=True,
            batch_size=args.bsz,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=True
        )
        memory_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                args.data_root, train=True, transform=single_transform, download=True
            ),
            shuffle=False,
            batch_size=args.bsz,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                args.data_root, train=False, transform=single_transform, download=True,
            ),
            shuffle=False,
            batch_size=args.bsz,
            pin_memory=True,
            num_workers=args.num_workers
        )
    elif args.pt_dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR100(
                args.data_root, train=True, transform=ContrastiveLearningTransform(args, simple=args.basic_aug), download=True
            ),
            shuffle=True,
            batch_size=args.bsz,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=True
        )
        memory_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR100(
                args.data_root, train=True, transform=single_transform, download=True
            ),
            shuffle=False,
            batch_size=args.bsz,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR100(
                args.data_root, train=False, transform=single_transform, download=True,
            ),
            shuffle=False,
            batch_size=args.bsz,
            pin_memory=True,
            num_workers=args.num_workers
        )


    # models
    if args.tfm != 'none':
        args.num_rotate_classes = num_tfm_classes[args.tfm]

    main_branch = Branch(args, encoder=encoder).cuda()

    if args.loss == 'simsiam':
        dim_proj = [int(x) for x in args.dim_proj.split(',')]
        predictor = PredictionMLP(dim_proj[1], args.dim_pred, dim_proj[1]).cuda()
    else:
        predictor = None

    # optimization
    if not args.BE:
        optimizer = torch.optim.SGD(
            main_branch.parameters(),
            momentum=0.9,
            lr=args.lr * args.bsz / 256,
            weight_decay=args.wd
        )
    else:
        my_list = ['alpha', 'gamma']
        params_multi_tmp= list(filter(lambda kv: (my_list[0] in kv[0]) or (my_list[1] in kv[0]) , main_branch.named_parameters()))
        param_core_tmp = list(filter(lambda kv: (my_list[0] not in kv[0]) and (my_list[1] not in kv[0]), main_branch.named_parameters()))

        params_multi=[param for name, param in params_multi_tmp]
        param_core=[param for name, param in param_core_tmp]
        # params_backbone = [param for name,param in ]
        print("Total slow weights params: {:.3f}M; Trainable: {:.3f}M".format(
                sum(p.numel() for p in param_core)/1e6, sum(p.numel() for p in param_core if p.requires_grad==True)/1e6))
        print("Total fast weights params: {:.3f}M; Trainable: {:.3f}M".format(
                sum(p.numel() for p in params_multi)/1e6, sum(p.numel() for p in params_multi if p.requires_grad==True)/1e6))
        optimizer = torch.optim.SGD(
            [{'params': param_core,'weight_decay': args.wd},
            {'params': params_multi, 'weight_decay': 0.0}],
            momentum=0.9,
            lr=args.lr * args.bsz / 256,
        )

    if args.loss == 'simsiam':
        pred_optimizer = torch.optim.SGD(
            predictor.parameters(),
            momentum=0.9,
            lr=args.lr * args.bsz / 256,
            weight_decay=args.wd
        )

    args.start_epoch = 1
    # automatically resume if checkpoint exists
    if args.auto_resume:
        if os.path.isfile(os.path.join(args.path_dir,'last.pth')):
            args.resume = os.path.join(args.path_dir,'last.pth')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            main_branch.load_state_dict(checkpoint['state_dict'])
            if predictor is not None:
                predictor.load_state_dict(checkpoint['pred_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            if checkpoint['epoch'] == args.epochs + 1:
                sys.exit() # run already completed

    # macros
    backbone = main_branch.encoder
    projector = main_branch.projector

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))
    scaler = GradScaler()

    # training
    for e in range(args.start_epoch, args.epochs + 1):
        # declaring train
        main_branch.train()
        if args.loss == 'simsiam':
            predictor.train()

        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # adjust
            lr = adjust_learning_rate(epochs=args.epochs,
                                      warmup_epochs=args.warmup_epochs,
                                      base_lr=args.lr * args.bsz / 256,
                                      optimizer=optimizer,
                                      loader=train_loader,
                                      step=it)
            # zero grad
            main_branch.zero_grad()
            if args.loss == 'simsiam':
                predictor.zero_grad()

            def forward_step():
                rot_loss = 0.
                loss = 0.
                x1 = inputs[0]
                x2 = inputs[1]
                if args.BE:
                    x1 = x1.repeat(args.num_be,1,1,1)
                    x2 = x2.repeat(args.num_be,1,1,1)

                x1 = x1.cuda()
                x2 = x2.cuda()
                if args.tfm_mode == 'inv':
                    x1 = choose_transform[args.tfm](x1, 0, number=1, trans_p=args.trans_p)
                    x2 = choose_transform[args.tfm](x2, 0, number=1, trans_p=args.trans_p)

                b1 = backbone(x1)
                b2 = backbone(x2)
                if args.sep_proj:
                    b1 = b1.chunk(args.num_be)
                    b2 = b2.chunk(args.num_be)
                z1 = projector(b1)
                z2 = projector(b2)

                # forward pass
                if args.loss == 'simclr':
                    if not args.BE:
                        loss = info_nce_loss(z1, z2) / 2 + info_nce_loss(z2, z1) / 2
                    else:
                        if args.sep_proj:
                            z1_list, z2_list = z1, z2
                        else:
                            z1_list = z1.chunk(args.num_be)
                            z2_list = z2.chunk(args.num_be)
                        for ii, (z11,z22) in enumerate(zip(z1_list,z2_list)):
                            loss += info_nce_loss(z11, z22) / 2 + info_nce_loss(z22, z11) / 2

                elif args.loss == 'simsiam':
                    if not args.BE:
                        p1 = predictor(z1)
                        p2 = predictor(z2)
                        loss = negative_cosine_similarity_loss(p1, z2) / 2 + negative_cosine_similarity_loss(p2, z1) / 2
                    else:
                        raise
                else:
                    raise

                # if args.lmbd > 0:
                if args.tfm_mode == 'eq':
                    rotated_images, rotated_labels = choose_transform[args.tfm](inputs[2], 0, number=args.num_eq, trans_p=args.trans_p)
                    b = backbone(rotated_images)
                    logits = main_branch.predictor2(b)
                    rot_loss = F.cross_entropy(logits, rotated_labels)
                    loss += args.lmbd * rot_loss
                return loss, rot_loss

            # optimization step
            if args.fp16:
                with autocast():
                    loss, rot_loss = forward_step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if args.loss == 'simsiam':
                    scaler.step(pred_optimizer)

            else:
                loss, rot_loss = forward_step()
                loss.backward()
                optimizer.step()
                if args.loss == 'simsiam':
                    pred_optimizer.step()

        if args.fp16:
            with autocast():
                knn_acc = knn_loop(backbone, memory_loader, test_loader, BE=args.BE, num_models=args.num_be)
        else:
            knn_acc = knn_loop(backbone, memory_loader, test_loader, BE=args.BE, num_models=args.num_be)


        line_to_print = (
            f'epoch: {e} | knn_acc: {knn_acc} | '
            f'loss: {loss.item():.3f} | rot_loss: {float(rot_loss):.3f} | lr: {lr:.6f} | '
            f'time_elapsed: {time.time() - start:.3f}'
        )
        if file_to_update:
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
        print(line_to_print)

        if e % args.save_every == 0:
            torch.save(dict(epoch=e, state_dict=main_branch.state_dict()),
                       os.path.join(args.path_dir, f'{e}.pth'))
        current_dict = dict(epoch=e, state_dict=main_branch.state_dict(), optimizer=optimizer.state_dict())
        if predictor is not None:
            current_dict['pred_state_dict'] = predictor.state_dict()
        torch.save(current_dict,
                   os.path.join(args.path_dir, f'last.pth'))

    return main_branch.encoder, file_to_update


def eval_loop(encoder, file_to_update, ind=None):
    # dataset
    if args.ft_dataset == 'cifar10':
        train_transform = T.Compose([
            T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        test_transform = T.Compose([
            T.Resize(36, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(32),
            T.ToTensor(),
            normalize
        ])

        train_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(args.data_root, train=True, transform=train_transform, download=True),
            shuffle=True,
            batch_size=256,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(args.data_root, train=False, transform=test_transform, download=True),
            shuffle=False,
            batch_size=256,
            pin_memory=True,
            num_workers=args.num_workers
        )

        classifier = nn.Linear(512, 10).cuda()
    elif args.ft_dataset == 'cifar100':
        train_transform = T.Compose([
            T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize_c100
        ])
        test_transform = T.Compose([
            T.Resize(36, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(32),
            T.ToTensor(),
            normalize_c100
        ])

        train_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR100(args.data_root, train=True, transform=train_transform, download=True),
            shuffle=True,
            batch_size=256,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR100(args.data_root, train=False, transform=test_transform, download=True),
            shuffle=False,
            batch_size=256,
            pin_memory=True,
            num_workers=args.num_workers
        )
        classifier = nn.Linear(512, 100).cuda()

    # optimization
    if args.ft_mode == 'lp':
        params = classifier.parameters()
    elif args.ft_mode == 'ft':
        params = list(classifier.parameters()) + list(encoder.parameters())

    optimizer = torch.optim.SGD(
        params,
        momentum=0.9,
        lr=args.ft_lr,
        weight_decay=0
    )
    scaler = GradScaler()

    # training
    for e in range(1, 101):
        # declaring train
        classifier.train()
        if args.ft_mode == 'lp':
            encoder.eval()
        elif args.ft_mode == 'ft':
            encoder.train()
        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # adjust
            adjust_learning_rate(epochs=100,
                                 warmup_epochs=0,
                                 base_lr=args.ft_lr,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            classifier.zero_grad()

            def forward_step():
                if args.ft_mode == 'lp':
                    with torch.no_grad():
                        b = encoder(inputs.cuda())
                elif args.ft_mode == 'ft':
                    b = encoder(inputs.cuda())

                logits = classifier(b)
                loss = F.cross_entropy(logits, y.cuda())
                return loss

            # optimization step
            if args.fp16:
                with autocast():
                    loss = forward_step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_step()
                loss.backward()
                optimizer.step()

        if e % 10 == 0:
            accs = []
            classifier.eval()
            encoder.eval()
            for idx, (images, labels) in enumerate(test_loader):
                with torch.no_grad():
                    if args.fp16:
                        with autocast():
                            b = encoder(images.cuda())
                            preds = classifier(b).argmax(dim=1)
                    else:
                        b = encoder(images.cuda())
                        preds = classifier(b).argmax(dim=1)
                    hits = (preds == labels.cuda()).sum().item()
                    accs.append(hits / b.shape[0])
            accuracy = np.mean(accs) * 100
            # final report of the accuracy
            line_to_print = (
                f'seed: {ind} | accuracy (%) @ epoch {e}: {accuracy:.2f}'
            )
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
            print(line_to_print)

    return accuracy


def main(args):
    fix_seed(args.seed)
    # logging
    username = getpass.getuser()
    args.path_dir = f'/home/gridsan/{username}/MAML-Soljacic/charlotte/{args.pt_dataset}_pt_ckpts/'

    os.makedirs(args.path_dir, exist_ok=True)
    args.path_dir = os.path.join(args.path_dir, args.exp_id)
    os.makedirs(args.path_dir, exist_ok=True)
    if args.server == 'aimos':
        make_sh_and_submit(args, delay=6)

    encoder, file_to_update = ssl_loop(args)
    accs = []
    for i in range(5):
        accs.append(eval_loop(copy.deepcopy(encoder), file_to_update, i))
    line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
    file_to_update.write(line_to_print + '\n')
    file_to_update.flush()
    print(line_to_print)


def make_sh_and_submit(args, delay=0):
    os.makedirs('./scripts/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    options = args.arg_str
    if delay == 0:
        name = ''.join([opt1.replace("--","").replace("=","") for opt1 in options.split(" ")])
        name = args.add_prefix + name

    else: # log_id should be already defined
        name = args.exp_id
    print('Submitting the job with options: ')
    # print(options)
    print(f"experiment name: {name}")

    if args.server == 'aimos':
        options += f'--server=aimos --arg_str=\"{args.arg_str}\" '
        preamble = (
            f'#!/bin/sh\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=20\n#SBATCH '
            f'-N 1\n#SBATCH -t 360\n#SBATCH ')
        preamble += f'--begin=now+{delay}hour\n#SBATCH '
        preamble += (f'-o ./logs/{name}.out\n#SBATCH '
                        f'--job-name={name}_{delay}\n#SBATCH '
                        f'--open-mode=append\n\n')

    else:
        import getpass
        username = getpass.getuser()
        options += f'--server={args.server} '
        preamble = (
            f'#!/bin/sh\n#SBATCH --gres=gpu:volta:1\n#SBATCH --cpus-per-task=20\n#SBATCH '
            f'-o ./logs/{name}.out\n#SBATCH '
            f'--job-name={name}\n#SBATCH '
            f'--open-mode=append\n\n'
        )
    with open(f'./scripts/{name}_{delay}.sh', 'w') as file:
        file.write(preamble)
        file.write("echo \"current time: $(date)\";\n")
        file.write(
            f'python {sys.argv[0]} '
            f'{options} --auto_resume --exp-id={name} '
        )
        # if args.server == 'sc' or args.server == 'rumensc':
            # file.write(f'--data_root=/home/gridsan/{username}/MAML-Soljacic/cifar_stl_data/ ')
        file.write(f'--data_root=/home/gridsan/groups/MAML-Soljacic/cifar_stl_data/ ')

    os.system(f'sbatch ./scripts/{name}_{delay}.sh')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_proj', default='2048,2048', type=str)
    parser.add_argument('--dim_pred', default=512, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--loss', default='simclr', type=str, choices=['simclr', 'simsiam'])
    parser.add_argument('--save_every', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--path_dir', default='./experiment', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lmbd', default=0.0, type=float)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--ft_dataset', default='cifar10', type=str)
    parser.add_argument('--ft_lr', default=30, type=float)
    parser.add_argument('--ft_mode', default='lp', type=str, choices=['lp','ft'])


    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--server', default='sc', type=str)

    parser.add_argument('--basic_aug', action='store_true')
    parser.add_argument('--tfm', type=str, default='none', choices=['none','stylize', 'jigsaw', 'rotate', 'blur', 'solarize', 'vflip', 'invert', 'grayscale','halfswap'])
    parser.add_argument('--tfm_mode', type=str, choices=['none','inv', 'eq'])
    parser.add_argument('--num_eq', type=int, default=24)
    parser.add_argument('--trans_p', default=0.5, type=float)
    parser.add_argument('--num_rotate_classes', default=4, type=int)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--arg_str', default='--', type=str)
    parser.add_argument('--add_prefix', default='', type=str)
    parser.add_argument('--exp-id', default='', type=str)

    parser.add_argument('--pt_dataset', default='cifar10', type=str)

    parser.add_argument('--BE', action='store_true')
    parser.add_argument('--num_be', default=3, type=int)
    parser.add_argument('--sep_proj', action='store_true')

    args = parser.parse_args()

    if args.submit:
        make_sh_and_submit(args)
    else:
        username = getpass.getuser()
        args.data_root = f'/home/gridsan/{username}/MAML-Soljacic/cifar_stl_data/'
        main(args)
