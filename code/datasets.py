import os
import pickle
import math
import random
import numpy as np
import glob
import pandas as pd
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import BatchSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, ImageOps, ImageFilter

from utils import GaussianBlur, Solarization, load_txt


try:
    from torchvision.datasets import Flowers102, SUN397, OxfordIIITPet, VOCDetection, StanfordCars, DTD, Caltech101, Food101, FGVCAircraft
except ImportError:
    from datasets_v08 import Flowers102, SUN397, OxfordIIITPet, Food101, StanfordCars, FGVCAircraft, DTD, Caltech101
try:
    from torchvision.datasets import INaturalist as iNat
except ImportError:
    from datasets_inat import INaturalist as iNat

dataset_norms = {
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'imagenet-a': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'imagenet-r': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'imagenet-v2': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'imagenet-sketch': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'imagenet-100': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'imagenet-100-a': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'imagenet-100-r': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'imagenet-100-v2': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'imagenet-100-sketch': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'flowers-102': ([0.5153, 0.4172, 0.3444], [0.2981, 0.2516, 0.2915]),
    'inat-1k': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #not calculated
    'cub-200': ([0.4821, 0.4905, 0.4241], [0.2289, 0.2249, 0.2596]),
    'objectnet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),#not calculated
    'cifar10': ([0.49139968,  0.48215841,  0.44653091], [0.24703223,  0.24348513,  0.26158784]),
    'cifar100': ([0.50707516,  0.48654887,  0.44091784], [0.26733429,  0.25643846,  0.27615047]),
    'ftcifar10': ([0.49139968,  0.48215841,  0.44653091], [0.24703223,  0.24348513,  0.26158784]),
    'ftcifar100': ([0.50707516,  0.48654887,  0.44091784], [0.26733429,  0.25643846,  0.27615047]),
    # 'cifar10': ([0.485, 0.456, 0.406], [0.228, 0.224, 0.225]),
    # 'cifar100': ([0.485, 0.456, 0.406], [0.228, 0.224, 0.225]),
    'cifar10-c': ([0.49139968,  0.48215841,  0.44653091], [0.24703223,  0.24348513,  0.26158784]),
    'cifar100-c': ([0.50707516,  0.48654887,  0.44091784], [0.26733429,  0.25643846,  0.27615047]),
    'food': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # not calculated
    'pets': ([0.4814, 0.4430, 0.3942], [0.2596, 0.2534, 0.2597]),
    'sun-397': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # not calculated
    'cars': ([0.4516, 0.4353, 0.4358], [0.2897, 0.2877, 0.2947]),
    'aircraft': ([0.4897, 0.5163, 0.5357], [0.2294, 0.2219, 0.2499]),
    'voc-2007': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # not calculated
    'dtd': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # not calculated
    'caltech-101': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

dataset_num_classes = {
    'imagenet': 1000,
    'imagenet-a': 1000,
    'imagenet-r': 1000,
    'imagenet-v2': 1000,
    'imagenet-sketch': 1000,
    'imagenet-100': 100,
    'imagenet-100-a': 100,
    'imagenet-100-r': 100,
    'imagenet-100-v2': 100,
    'imagenet-100-sketch': 100,
    'flowers-102': 102,
    'inat-1k': 1010,
    'cub-200': 200,
    'cifar10': 10,
    'cifar100': 100,
    'ftcifar10': 10,
    'ftcifar100': 100,
    'cifar10-c': 10,
    'cifar100-c': 100,
    'food': 101,
    'pets': 37,
    'sun-397': 397,
    'cars': 196,
    'aircraft': 100,
    'voc-2007': 20,
    'dtd': 47,
    'caltech-101': 101,
    # 'birdsnap':
}

dataset_img_sizes = {
    'imagenet': 224,
    'imagenet-a': 224,
    'imagenet-r': 224,
    'imagenet-v2': 224,
    'imagenet-sketch': 224,
    'imagenet-100': 224,
    'imagenet-100-a': 224,
    'imagenet-100-r': 224,
    'imagenet-100-v2': 224,
    'imagenet-100-sketch': 224,
    'flowers-102': 224, #check
    'inat-1k': 224,
    'cub-200': 224, #check
    'cifar10': 32,
    'cifar100': 32,
    'ftcifar10': 32,
    'ftcifar100': 32,
    'cifar10-c': 32,
    'cifar100-c': 32,
    'food': 224,
    'pets': 224,
    'sun-397': 224, #check
    'cars': 224,
    'aircraft': 224,
    'voc-2007': 224,
    'dtd': 224, #check
    'caltech-101': 224,
    # 'birdsnap':
}

class INaturalist(iNat):
    def __init__(self, root, version='2019', target_type='full', transform=None, target_transform=None, download=False):
        super().__init__(root, version, target_type, transform, target_transform, download)

    def __getitem__(self, index):
        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname)).convert('RGB')

        target = []
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

# Dataset
class Stylized_Dataset(Dataset):
    def __init__(self, root, transform=None, split='train', version='inv', p=0.5):
        self.img_path = []
        self.labels = []
        # self.style_labels = []
        self.transform = transform
        self.version = version
        self.img_path = list(np.load(f"{root}/sty_IN_{split}_img_paths.npy", allow_pickle=True))
        self.labels = list(np.load(f"{root}/sty_IN_{split}_labels.npy", allow_pickle=True))
        self.p = p

        # img_dict = {}
        # with open(txt) as f:
        #     for line in f:
        #         img1 = line.split()[0]
        #         label = int(line.split()[1])
        #         sty_label = int(line.split()[2])
        #         img_dict[sty_label] = img1
        #         if sty_label == 4:
        #             self.img_path.append(img_dict)
        #             self.labels.append(label)
        #             img_dict = {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]

        if self.version == 'inv':
            style = np.random.choice(5, 1, p=[(1-self.p), self.p/4, self.p/4, self.p/4, self.p/4]).item()
            # style = np.random.choice(5)
            path = self.img_path[index][style]
            # path = '/home/cloh/imagenet_data/train/n04596742/n04596742_536.JPEG'

            with open(path, 'rb') as f:
                sample = Image.open(f).convert('RGB')
            if self.transform is not None:
                sample = self.transform(sample)
            # return sample, label, style

            return sample, label


        elif self.version == 'eq':

            for sty in range(5):
                path = self.img_path[index][sty]
                # path = '/home/cloh/imagenet_data/train/n04596742/n04596742_536.JPEG'
                with open(path, 'rb') as f:
                    sample = Image.open(f).convert('RGB')
                if self.transform is not None:
                    sample = self.transform(sample)
                if sty == 0:
                    samples = sample.unsqueeze(0)
                else:
                    samples = torch.cat([samples, sample.unsqueeze(0)], dim=0)

            return samples, label


def build_dataset(args):
    if args.mask_mode == 'topk' or args.mask_mode == 'topk_sum' or args.mask_mode == 'topk_agg_sum':
        dataset = TopKDataset(args.dataset, args.data / 'train', args.topk_path, args.topk, Transform(args))
    elif args.mask_mode == 'weight_anchor_logits' or args.mask_mode == 'weight_class_logits':
        dataset = LogitsDataset(args.dataset, args.data / 'train', args.topk_path, Transform(args))

    # if args.stylize == 'eq':
    #     dataset = Stylized_Dataset(args.data, transform=SingleTransform(args), split='train',version=args.stylize, p=args.trans_p)
    # elif args.stylize == 'inv':
    #     dataset = Stylized_Dataset(args.data, transform=Transform(args), split='train',version=args.stylize, p=args.trans_p)

    else:
        if args.dataset == 'imagenet':
            # using x % of the training data
            if args.training_ratio < 1:
                dataset = ImageFolder(args.data / 'train', Transform(args))
                split_idx_path = f'./misc/imagenet-1k-training-ratio-{args.training_ratio}-idxs.pth'
                
                # creating/loading train-val (90 - 10) split
                if os.path.exists(split_idx_path):
                    split_idx = torch.load(split_idx_path)
                else:
                    # organize indices of each class into a dict
                    class_to_idx = classes_to_idxs(dataset)

                    split_idx = {}
                    train_idx = []

                    # sample 10% of data from each class
                    for c in class_to_idx.keys():
                        chosen_idx = random.sample(class_to_idx[c], int(len(class_to_idx[c]) * args.training_ratio))
                        train_idx += chosen_idx

                    split_idx['train'] = train_idx

                    # save
                    torch.save(split_idx, split_idx_path)

                dataset = torch.utils.data.Subset(dataset, split_idx['train'])
            
            else:
                dataset = ImageFolder(args.data / 'train', Transform(args))

            # if not args.use_smaller_split:
            #     # creating/loading train-val (50 images per class in validation) split
            #     split_idx_path = './misc/imagenet_train_val_split.pth'
            #     dataset = ImageFolder(args.data / 'train', Transform(args))

            #     if os.path.exists(split_idx_path):
            #         split_idx = torch.load(split_idx_path)
            #     else:
            #         # organize indices of each class into a dict
            #         class_to_idx = classes_to_idxs(dataset)

            #         split_idx = {}

            #         for i in range(10):
            #             split_idx[i] = {}
            #             all_idx = set(range(len(dataset)))
            #             val_idx = []

            #             # sample 50 images from each class
            #             for c in class_to_idx.keys():
            #                 chosen_idx = random.sample(class_to_idx[c], 50)
            #                 val_idx += chosen_idx

            #             val_idx = set(val_idx)
            #             train_idx = all_idx - val_idx

            #             split_idx[i]['train'] = list(train_idx)
            #             split_idx[i]['val'] = list(val_idx)

            #         torch.save(split_idx, split_idx_path)

            #     if args.train_val_split > -1:
            #         dataset = Subset(dataset, split_idx[args.train_val_split]['train'])
            # else:
            #     dataset = Split_Dataset(args.data,  \
            #             f'./calib_splits/am_IN{args.val_perc}_train.txt',
            #             transform=Transform(args))
            #     print(f"loading ./calib_splits/am_IN{args.val_perc}_train.txt with Contrastive Transforms")

        elif args.dataset == 'cifar100':
            dataset = CIFAR100(args.data, transform=Transform(args), train=True)

    return dataset

def build_dataloaders(args):
    print('Building dataset & dataloaders')

    train_dataset = None
    val_dataset = None
    test_dataset = None
    train_loader = None
    test_dataset = None

    # dataset img size
    img_size = dataset_img_sizes[args.dataset]
    # normalize
    normalize = transforms.Normalize(dataset_norms[args.dataset][0], dataset_norms[args.dataset][1])
    # transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(img_size+32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])

    if ('cifar' in args.dataset and 'ftcifar' not in args.dataset) or args.dataset == 'caltech-101' or args.dataset == 'voc-2007':
        train_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    if args.eval_mode == 'log_reg':
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    # elif args.eval_mode == 'temp_scale':
    #     train_transforms = transforms.Compose([
    #         transforms.Resize(img_size+32),
    #         transforms.CenterCrop(img_size),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    #
    #     val_transforms = transforms.Compose([
    #         transforms.Resize(img_size+32),
    #         transforms.CenterCrop(img_size),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    print(f'Creating {args.dataset} dataset')
    # train & valid datasets
    if args.dataset in {'imagenet', 'imagenet-100'} and not (args.eval_subset100 or args.eval_var_subset is not None):

        if 'sc' in args.server or args.server == 'aimos':
            traindir = args.data / 'train'
            testdir = args.data / 'val'
        else:
            traindir = args.data / f'{args.dataset}/train'
            testdir = args.data / f'{args.dataset}/val'

        if args.use_smaller_split_val:
            train_dataset = Split_Dataset(args.data,  \
                    f'./calib_splits/am_{args.dataset}_{args.val_perc}percent_val.txt',
                    transform=train_transforms)
            print(f'Loading ./calib_splits/am_{args.dataset}_{args.val_perc}percent_val.txt')

        elif args.use_smaller_split:
            # train_dataset = Split_Dataset(args.data,  \
            #         f'./calib_splits/am_{args.dataset}_{args.val_perc}percent_train.txt',
            #         transform=train_transforms)
            train_dataset = Split_Dataset(args.data,  \
                    f'./calib_splits/am_IN{args.val_perc}_train.txt',
                    transform=train_transforms)
            print(f'Loading ./calib_splits/am_IN{args.val_perc}_train.txt')

        elif args.eval_mode == 'temp_scale':
            if args.train_val_split > -1:
                train_dataset = ImageFolder(traindir, transform=val_transforms)
                print(f'Train dataset: Loading {traindir} with val transforms')
                print(f"Loading train val split = {args.train_val_split}")
                split_idx_path = './misc/imagenet_train_val_split.pth'
                if os.path.exists(split_idx_path):
                    split_idx = torch.load(split_idx_path)
                else:
                    raise Exception('Imagenet train-val split dict file does not exist! git pull ')
                train_dataset = Subset(train_dataset, split_idx[args.train_val_split]['val'])
                print(f"Total train: {len(train_dataset)} ")
            else:
                train_dataset = Split_Dataset(args.data,  \
                        f'./calib_splits/am_{args.dataset}_{args.val_perc}percent_val.txt',
                        transform=val_transforms)
                print(f'Loading ./calib_splits/am_{args.dataset}_{args.val_perc}percent_val.txt with val transforms')

        elif args.fold is not None:
            train_dataset = Split_Dataset(args.data,  \
                    f'./calib_splits/am_{args.dataset}_5uniqfolds_train_fold{args.fold}.txt',
                    transform=train_transforms)
            print(f'Loading ./calib_splits/am_{args.dataset}_5uniqfolds_train_fold{args.fold}.txt with train transforms')

        else:
            train_dataset = ImageFolder(traindir, transform=train_transforms)
            print(f'Train dataset: Loading {traindir} with train transforms')

        if args.fold is not None:
            test_dataset = Split_Dataset(args.data,  \
                    f'./calib_splits/am_{args.dataset}_5uniqfolds_val_fold{args.fold}.txt',
                    transform=val_transforms)
            print(f'Loading ./calib_splits/am_{args.dataset}_5uniqfolds_val_fold{args.fold}.txt with val transforms')

        else:
            if (args.train_val_split > -1 and args.eval_mode != 'temp_scale'):
                test_dataset = None # set train_val_split to -1 for final test
            else:
                if args.use_smaller_split:
                    test_dataset = Split_Dataset(args.data,  \
                            f'./calib_splits/am_IN{args.val_perc}_val.txt',
                            transform=val_transforms)
                    print(f'Loading ./calib_splits/am_IN{args.val_perc}_val.txt')
                else:
                    test_dataset = ImageFolder(testdir, transform=val_transforms) # val set is always the same
                    print(f'Test dataset: Loading {testdir} with val transforms')

        if args.eval_on_train:
            test_dataset = train_dataset
            test_dataset.transform = val_transforms


        # creating/loading train-val (50 images per class in validation) split
        if args.train_val_split > -1 and args.eval_mode != 'temp_scale':
            val_dataset = ImageFolder(traindir, transform=val_transforms)
            print(f'Val dataset: Loading {traindir} with val transforms')

            print(f"Loading train val split = {args.train_val_split}")
            split_idx_path = './misc/imagenet_train_val_split.pth'

            if os.path.exists(split_idx_path):
                split_idx = torch.load(split_idx_path)
            else:
                raise Exception('Imagenet train-val split dict file does not exist! git pull ')

        # if args.train_val_split > -1:
            train_dataset = Subset(train_dataset, split_idx[args.train_val_split]['train'])
            val_dataset = Subset(val_dataset, split_idx[args.train_val_split]['val'])
            print(f"Total train: {len(train_dataset)}, val: {len(val_dataset)} ")

        else:
            val_dataset = None
            print(f"Total train: {len(train_dataset)}, test: {len(test_dataset)}")

        # if args.train_percent in {1, 10}:
        #     train_dataset.samples = []
        #     for fname in args.train_files:
        #         fname = fname.decode().strip()
        #         cls = fname.split('_')[0]
        #         train_dataset.samples.append(
        #             (traindir / cls / fname, train_dataset.class_to_idx[cls]))
    elif args.eval_subset100:
        train_dataset = Split_Dataset(args.data,  \
                f'./calib_splits/IN100_with_INlabels_train.txt',
                transform=train_transforms)
        print("loading IN100_with_INlabels_train.txt")
        if args.eval_on_train:
            test_dataset = train_dataset
            test_dataset.transform = val_transforms
        else:
            test_dataset = Split_Dataset(args.data,  \
                f'./calib_splits/IN100_with_INlabels_val.txt',
                transform=val_transforms)
            print("loading IN100_with_INlabels_val.txt")

        print(f"Total train: {len(train_dataset)}, val: {len(test_dataset)}")

    elif args.eval_var_subset is not None:
        train_dataset = Split_Dataset(args.data,  \
                f'./calib_splits/am_IN{args.eval_var_subset}_with_INlabels_train.txt',
                transform=train_transforms)
        print(f"Loading ./calib_splits/am_IN{args.eval_var_subset}_with_INlabels_train.txt")
        if args.eval_on_train:
            test_dataset = train_dataset
            test_dataset.transform = val_transforms

        else:
            test_dataset = Split_Dataset(args.data,  \
                f'./calib_splits/am_IN{args.eval_var_subset}_with_INlabels_val.txt',
                transform=val_transforms)
            print(f"Loading ./calib_splits/am_IN{args.eval_var_subset}_with_INlabels_val.txt")


        print(f"Total train: {len(train_dataset)}, val: {len(test_dataset)}")

    elif args.dataset in {'imagenet-v2', 'imagenet-100-v2'}:
        v2_dataset_names = ['imagenetv2-matched-frequency-format-val', 'imagenetv2-threshold0.7-format-val', 'imagenetv2-top-images-format-val']
        v2_datasets = [ImageFolder(args.data / f'{args.dataset}/{n}', transform=val_transforms) for n in v2_dataset_names]
        val_dataset = ConcatDataset(v2_datasets)

    elif args.dataset in {'imagenet-a', 'imagenet-100-a', 'imagenet-sketch', 'imagenet-100-sketch', 'imagenet-r', 'imagenet-100-r'}:
        val_dataset = ImageFolder(args.data / f'{args.dataset}', val_transforms)

    elif args.dataset == 'flowers-102':
        train_dataset = Flowers102(args.data, split='train', transform=train_transforms)
        val_dataset = Flowers102(args.data, split='test', transform=val_transforms) # may need to change to test
        print("loading flowers102")

    elif args.dataset == 'inat-1k':
        dataset = INaturalist(args.data, version='2019', transform=train_transforms)
        split_idx_path = './misc/inat-1k-train-val-split-idx.pth'

        # creating/loading train-val (90 - 10) split
        if os.path.exists(split_idx_path):
            split_idx = torch.load(split_idx_path)
        else:
            # organize indices of each class into a dict
            class_to_idx = classes_to_idxs(dataset)

            split_idx = {}
            all_idx = set(range(len(dataset)))
            val_idx = []

            # sample 10% of data from each class
            for c in class_to_idx.keys():
                chosen_idx = random.sample(class_to_idx[c], len(class_to_idx[c]) // 10)
                val_idx += chosen_idx
            val_idx = set(val_idx)
            train_idx = all_idx - val_idx

            split_idx['train'] = list(train_idx)
            split_idx['val'] = list(val_idx)

            # save
            torch.save(split_idx, split_idx_path)

        train_dataset = Subset(dataset, split_idx['train'])
        train_dataset.classes = dataset.all_categories
        val_dataset = Subset(dataset, split_idx['val'])
        val_dataset.classes = dataset.all_categories

    elif args.dataset == 'cub-200':
        train_dataset = Cub2011(args.data / 'cub-200', train=True, transform=train_transforms)
        val_dataset = Cub2011(args.data / 'cub-200', train=False, transform=val_transforms)

    elif args.dataset == 'objectnet':
        val_dataset = ObjectNetDataset(args.data, transform=val_transforms)

    elif args.dataset == 'cifar10' or args.dataset == 'ftcifar10':
        train_dataset = CIFAR10(args.data, train=True, transform=train_transforms, download=True)
        val_dataset = CIFAR10(args.data, train=False, transform=val_transforms, download=True)

    elif args.dataset == 'cifar100' or args.dataset == 'ftcifar100':
        train_dataset = CIFAR100(args.data, train=True, transform=train_transforms, download=True)
        val_dataset = CIFAR100(args.data, train=False, transform=val_transforms, download=True)

    elif args.dataset == 'cifar10-c':
        corruptions = load_txt('./misc/cifar10-c-corruptions.txt')
        datasets_c = []
        for corruption in corruptions:
            datasets_c.append(CIFAR10C(args.data, corruption, transform=val_transforms))

        val_dataset = ConcatDataset(datasets_c)

    elif args.dataset == 'cifar100-c':
        corruptions = load_txt('./misc/cifar10-c-corruptions.txt')
        datasets_c = []
        for corruption in corruptions:
            datasets_c.append(CIFAR10C(args.data, corruption, transform=val_transforms))

        val_dataset = ConcatDataset(datasets_c)

    elif args.dataset == 'food':
        train_dataset = Food101(args.data, split='train', transform=train_transforms, download=True)
        val_dataset = Food101(args.data, split='test', transform=val_transforms, download=True)

    elif args.dataset == 'pets':
        train_dataset = OxfordIIITPet(args.data, split='trainval', target_types='category', transform=train_transforms)
        val_dataset = OxfordIIITPet(args.data, split='test', target_types='category', transform=val_transforms)

    elif args.dataset == 'sun-397':
        dataset = SUN397(args.data, transform=train_transforms, download=True)
        split_idx_path = './misc/sun397-train-val-split-idx.pth'

        if os.path.exists(split_idx_path):
            split_idx = torch.load(split_idx_path)
        else:
            # # organize indices of each class into a dict
            # class_to_idx = classes_to_idxs(dataset)

            # split_idx = {}
            # train_idxs = []
            # val_idxs = []

            # # sample 10% of data from each class
            # for c in class_to_idx.keys():
            #     chosen_idx = random.sample(class_to_idx[c], 100)
            #     random.shuffle(chosen_idx)
            #     train_idxs += chosen_idx[:50]
            #     val_idxs += chosen_idx[50:]
            
            # print('Processing #1 partition of train/val')
            split_idx = {}
            train_idxs = []
            val_idxs = []
            train_partition = load_txt('./misc/sun397_training_01.txt')
            val_partition = load_txt('./misc/sun397_testing_01.txt')

            for i in range(len(dataset._image_files)):
                image_file = str(dataset._image_files[i])
                image_file = '/' + image_file[image_file.rfind('SUN397/')+len('SUN397/'):]

                if image_file in train_partition:
                    train_idxs.append(i)
                elif image_file in val_partition:
                    val_idxs.append(i)

            split_idx['train'] = train_idxs
            split_idx['val'] = val_idxs

            torch.save(split_idx, split_idx_path)

        train_dataset = Subset(dataset, split_idx['train'])
        val_dataset = Subset(dataset, split_idx['val'])
        print('dataset sizes: ', len(train_dataset), len(val_dataset))

    elif args.dataset == 'cars':
        train_dataset = StanfordCars(args.data, split='train', transform=train_transforms, download=True)
        val_dataset = StanfordCars(args.data, split='test', transform=val_transforms, download=True)

    # TODO: mean per-class accuracy for FGVC Aircraft, Oxford-IIIT Pets, Caltech-101, and Oxford 102 Flowers
    elif args.dataset == 'aircraft':
        train_dataset = FGVCAircraft(args.data, split='train', transform=train_transforms)
        val_dataset = FGVCAircraft(args.data, split='test', transform=val_transforms)

    # TODO: 11-point mAP metric
    elif args.dataset == 'voc-2007':
        train_dataset = VOCDetection(args.data, year='2007', split='train', transform=train_transforms)
        val_dataset = VOCDetection(args.data, year='2007', split='test', transform=val_transforms)

    elif args.dataset == 'dtd':
        train_dataset = DTD(args.data, image_set='train', partition=1, transform=train_transforms)
        val_dataset = DTD(args.data, image_set='test', partition=1, transform=val_transforms)

    elif args.dataset == 'caltech-101':
        dataset = Caltech101(args.data, target_type='category', transform=train_transforms)

        class_to_idx = classes_to_idxs(dataset)

        split_idx = {}
        all_idx = set(range(len(dataset)))
        val_idx = []

        # sample 10% of data from each class
        for c in class_to_idx.keys():
            chosen_idx = random.sample(class_to_idx[c], 30)
            val_idx += chosen_idx
        val_idx = set(val_idx)
        train_idx = all_idx - val_idx

        split_idx['train'] = list(train_idx)
        split_idx['val'] = list(val_idx)

        train_dataset = Subset(dataset, split_idx['train'])
        val_dataset = Subset(dataset, split_idx['val'])
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not supported.')

    train_kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.workers, pin_memory=True)
    val_kwargs = dict(train_kwargs)
    val_kwargs['batch_size'] = args.val_batch_size
    # val_kwargs['shuffle'] = False
    val_kwargs['shuffle'] = True


    # sampler & dataloader
    if train_dataset is not None:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs) if val_dataset else None
    test_loader = torch.utils.data.DataLoader(test_dataset, **val_kwargs) if test_dataset else None

    # setting up indices if dataset only contains subset of 1000 classes
    if args.dataset == 'imagenet-a':
        thousand_k_to_200 = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 0, 7: -1, 8: -1, 9: -1, 10: -1, 11: 1, 12: -1, 13: 2, 14: -1, 15: 3, 16: -1, 17: 4, 18: -1, 19: -1, 20: -1, 21: -1, 22: 5, 23: 6, 24: -1, 25: -1, 26: -1, 27: 7, 28: -1, 29: -1, 30: 8, 31: -1, 32: -1, 33: -1, 34: -1, 35: -1, 36: -1, 37: 9, 38: -1, 39: 10, 40: -1, 41: -1, 42: 11, 43: -1, 44: -1, 45: -1, 46: -1, 47: 12, 48: -1, 49: -1, 50: 13, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: 14, 58: -1, 59: -1, 60: -1, 61: -1, 62: -1, 63: -1, 64: -1, 65: -1, 66: -1, 67: -1, 68: -1, 69: -1, 70: 15, 71: 16, 72: -1, 73: -1, 74: -1, 75: -1, 76: 17, 77: -1, 78: -1, 79: 18, 80: -1, 81: -1, 82: -1, 83: -1, 84: -1, 85: -1, 86: -1, 87: -1, 88: -1, 89: 19, 90: 20, 91: -1, 92: -1, 93: -1, 94: 21, 95: -1, 96: 22, 97: 23, 98: -1, 99: 24, 100: -1, 101: -1, 102: -1, 103: -1, 104: -1, 105: 25, 106: -1, 107: 26, 108: 27, 109: -1, 110: 28, 111: -1, 112: -1, 113: 29, 114: -1, 115: -1, 116: -1, 117: -1, 118: -1, 119: -1, 120: -1, 121: -1, 122: -1, 123: -1, 124: 30, 125: 31, 126: -1, 127: -1, 128: -1, 129: -1, 130: 32, 131: -1, 132: 33, 133: -1, 134: -1, 135: -1, 136: -1, 137: -1, 138: -1, 139: -1, 140: -1, 141: -1, 142: -1, 143: 34, 144: 35, 145: -1, 146: -1, 147: -1, 148: -1, 149: -1, 150: 36, 151: 37, 152: -1, 153: -1, 154: -1, 155: -1, 156: -1, 157: -1, 158: -1, 159: -1, 160: -1, 161: -1, 162: -1, 163: -1, 164: -1, 165: -1, 166: -1, 167: -1, 168: -1, 169: -1, 170: -1, 171: -1, 172: -1, 173: -1, 174: -1, 175: -1, 176: -1, 177: -1, 178: -1, 179: -1, 180: -1, 181: -1, 182: -1, 183: -1, 184: -1, 185: -1, 186: -1, 187: -1, 188: -1, 189: -1, 190: -1, 191: -1, 192: -1, 193: -1, 194: -1, 195: -1, 196: -1, 197: -1, 198: -1, 199: -1, 200: -1, 201: -1, 202: -1, 203: -1, 204: -1, 205: -1, 206: -1, 207: 38, 208: -1, 209: -1, 210: -1, 211: -1, 212: -1, 213: -1, 214: -1, 215: -1, 216: -1, 217: -1, 218: -1, 219: -1, 220: -1, 221: -1, 222: -1, 223: -1, 224: -1, 225: -1, 226: -1, 227: -1, 228: -1, 229: -1, 230: -1, 231: -1, 232: -1, 233: -1, 234: 39, 235: 40, 236: -1, 237: -1, 238: -1, 239: -1, 240: -1, 241: -1, 242: -1, 243: -1, 244: -1, 245: -1, 246: -1, 247: -1, 248: -1, 249: -1, 250: -1, 251: -1, 252: -1, 253: -1, 254: 41, 255: -1, 256: -1, 257: -1, 258: -1, 259: -1, 260: -1, 261: -1, 262: -1, 263: -1, 264: -1, 265: -1, 266: -1, 267: -1, 268: -1, 269: -1, 270: -1, 271: -1, 272: -1, 273: -1, 274: -1, 275: -1, 276: -1, 277: 42, 278: -1, 279: -1, 280: -1, 281: -1, 282: -1, 283: 43, 284: -1, 285: -1, 286: -1, 287: 44, 288: -1, 289: -1, 290: -1, 291: 45, 292: -1, 293: -1, 294: -1, 295: 46, 296: -1, 297: -1, 298: 47, 299: -1, 300: -1, 301: 48, 302: -1, 303: -1, 304: -1, 305: -1, 306: 49, 307: 50, 308: 51, 309: 52, 310: 53, 311: 54, 312: -1, 313: 55, 314: 56, 315: 57, 316: -1, 317: 58, 318: -1, 319: 59, 320: -1, 321: -1, 322: -1, 323: 60, 324: 61, 325: -1, 326: 62, 327: 63, 328: -1, 329: -1, 330: 64, 331: -1, 332: -1, 333: -1, 334: 65, 335: 66, 336: 67, 337: -1, 338: -1, 339: -1, 340: -1, 341: -1, 342: -1, 343: -1, 344: -1, 345: -1, 346: -1, 347: 68, 348: -1, 349: -1, 350: -1, 351: -1, 352: -1, 353: -1, 354: -1, 355: -1, 356: -1, 357: -1, 358: -1, 359: -1, 360: -1, 361: 69, 362: -1, 363: 70, 364: -1, 365: -1, 366: -1, 367: -1, 368: -1, 369: -1, 370: -1, 371: -1, 372: 71, 373: -1, 374: -1, 375: -1, 376: -1, 377: -1, 378: 72, 379: -1, 380: -1, 381: -1, 382: -1, 383: -1, 384: -1, 385: -1, 386: 73, 387: -1, 388: -1, 389: -1, 390: -1, 391: -1, 392: -1, 393: -1, 394: -1, 395: -1, 396: -1, 397: 74, 398: -1, 399: -1, 400: 75, 401: 76, 402: 77, 403: -1, 404: 78, 405: -1, 406: -1, 407: 79, 408: -1, 409: -1, 410: -1, 411: 80, 412: -1, 413: -1, 414: -1, 415: -1, 416: 81, 417: 82, 418: -1, 419: -1, 420: 83, 421: -1, 422: -1, 423: -1, 424: -1, 425: 84, 426: -1, 427: -1, 428: 85, 429: -1, 430: 86, 431: -1, 432: -1, 433: -1, 434: -1, 435: -1, 436: -1, 437: 87, 438: 88, 439: -1, 440: -1, 441: -1, 442: -1, 443: -1, 444: -1, 445: 89, 446: -1, 447: -1, 448: -1, 449: -1, 450: -1, 451: -1, 452: -1, 453: -1, 454: -1, 455: -1, 456: 90, 457: 91, 458: -1, 459: -1, 460: -1, 461: 92, 462: 93, 463: -1, 464: -1, 465: -1, 466: -1, 467: -1, 468: -1, 469: -1, 470: 94, 471: -1, 472: 95, 473: -1, 474: -1, 475: -1, 476: -1, 477: -1, 478: -1, 479: -1, 480: -1, 481: -1, 482: -1, 483: 96, 484: -1, 485: -1, 486: 97, 487: -1, 488: 98, 489: -1, 490: -1, 491: -1, 492: 99, 493: -1, 494: -1, 495: -1, 496: 100, 497: -1, 498: -1, 499: -1, 500: -1, 501: -1, 502: -1, 503: -1, 504: -1, 505: -1, 506: -1, 507: -1, 508: -1, 509: -1, 510: -1, 511: -1, 512: -1, 513: -1, 514: 101, 515: -1, 516: 102, 517: -1, 518: -1, 519: -1, 520: -1, 521: -1, 522: -1, 523: -1, 524: -1, 525: -1, 526: -1, 527: -1, 528: 103, 529: -1, 530: 104, 531: -1, 532: -1, 533: -1, 534: -1, 535: -1, 536: -1, 537: -1, 538: -1, 539: 105, 540: -1, 541: -1, 542: 106, 543: 107, 544: -1, 545: -1, 546: -1, 547: -1, 548: -1, 549: 108, 550: -1, 551: -1, 552: 109, 553: -1, 554: -1, 555: -1, 556: -1, 557: 110, 558: -1, 559: -1, 560: -1, 561: 111, 562: 112, 563: -1, 564: -1, 565: -1, 566: -1, 567: -1, 568: -1, 569: 113, 570: -1, 571: -1, 572: 114, 573: 115, 574: -1, 575: 116, 576: -1, 577: -1, 578: -1, 579: 117, 580: -1, 581: -1, 582: -1, 583: -1, 584: -1, 585: -1, 586: -1, 587: -1, 588: -1, 589: 118, 590: -1, 591: -1, 592: -1, 593: -1, 594: -1, 595: -1, 596: -1, 597: -1, 598: -1, 599: -1, 600: -1, 601: -1, 602: -1, 603: -1, 604: -1, 605: -1, 606: 119, 607: 120, 608: -1, 609: 121, 610: -1, 611: -1, 612: -1, 613: -1, 614: 122, 615: -1, 616: -1, 617: -1, 618: -1, 619: -1, 620: -1, 621: -1, 622: -1, 623: -1, 624: -1, 625: -1, 626: 123, 627: 124, 628: -1, 629: -1, 630: -1, 631: -1, 632: -1, 633: -1, 634: -1, 635: -1, 636: -1, 637: -1, 638: -1, 639: -1, 640: 125, 641: 126, 642: 127, 643: 128, 644: -1, 645: -1, 646: -1, 647: -1, 648: -1, 649: -1, 650: -1, 651: -1, 652: -1, 653: -1, 654: -1, 655: -1, 656: -1, 657: -1, 658: 129, 659: -1, 660: -1, 661: -1, 662: -1, 663: -1, 664: -1, 665: -1, 666: -1, 667: -1, 668: 130, 669: -1, 670: -1, 671: -1, 672: -1, 673: -1, 674: -1, 675: -1, 676: -1, 677: 131, 678: -1, 679: -1, 680: -1, 681: -1, 682: 132, 683: -1, 684: 133, 685: -1, 686: -1, 687: 134, 688: -1, 689: -1, 690: -1, 691: -1, 692: -1, 693: -1, 694: -1, 695: -1, 696: -1, 697: -1, 698: -1, 699: -1, 700: -1, 701: 135, 702: -1, 703: -1, 704: 136, 705: -1, 706: -1, 707: -1, 708: -1, 709: -1, 710: -1, 711: -1, 712: -1, 713: -1, 714: -1, 715: -1, 716: -1, 717: -1, 718: -1, 719: 137, 720: -1, 721: -1, 722: -1, 723: -1, 724: -1, 725: -1, 726: -1, 727: -1, 728: -1, 729: -1, 730: -1, 731: -1, 732: -1, 733: -1, 734: -1, 735: -1, 736: 138, 737: -1, 738: -1, 739: -1, 740: -1, 741: -1, 742: -1, 743: -1, 744: -1, 745: -1, 746: 139, 747: -1, 748: -1, 749: 140, 750: -1, 751: -1, 752: 141, 753: -1, 754: -1, 755: -1, 756: -1, 757: -1, 758: 142, 759: -1, 760: -1, 761: -1, 762: -1, 763: 143, 764: -1, 765: 144, 766: -1, 767: -1, 768: 145, 769: -1, 770: -1, 771: -1, 772: -1, 773: 146, 774: 147, 775: -1, 776: 148, 777: -1, 778: -1, 779: 149, 780: 150, 781: -1, 782: -1, 783: -1, 784: -1, 785: -1, 786: 151, 787: -1, 788: -1, 789: -1, 790: -1, 791: -1, 792: 152, 793: -1, 794: -1, 795: -1, 796: -1, 797: 153, 798: -1, 799: -1, 800: -1, 801: -1, 802: 154, 803: 155, 804: 156, 805: -1, 806: -1, 807: -1, 808: -1, 809: -1, 810: -1, 811: -1, 812: -1, 813: 157, 814: -1, 815: 158, 816: -1, 817: -1, 818: -1, 819: -1, 820: 159, 821: -1, 822: -1, 823: 160, 824: -1, 825: -1, 826: -1, 827: -1, 828: -1, 829: -1, 830: -1, 831: 161, 832: -1, 833: 162, 834: -1, 835: 163, 836: -1, 837: -1, 838: -1, 839: 164, 840: -1, 841: -1, 842: -1, 843: -1, 844: -1, 845: 165, 846: -1, 847: 166, 848: -1, 849: -1, 850: 167, 851: -1, 852: -1, 853: -1, 854: -1, 855: -1, 856: -1, 857: -1, 858: -1, 859: 168, 860: -1, 861: -1, 862: 169, 863: -1, 864: -1, 865: -1, 866: -1, 867: -1, 868: -1, 869: -1, 870: 170, 871: -1, 872: -1, 873: -1, 874: -1, 875: -1, 876: -1, 877: -1, 878: -1, 879: 171, 880: 172, 881: -1, 882: -1, 883: -1, 884: -1, 885: -1, 886: -1, 887: -1, 888: 173, 889: -1, 890: 174, 891: -1, 892: -1, 893: -1, 894: -1, 895: -1, 896: -1, 897: 175, 898: -1, 899: -1, 900: 176, 901: -1, 902: -1, 903: -1, 904: -1, 905: -1, 906: -1, 907: 177, 908: -1, 909: -1, 910: -1, 911: -1, 912: -1, 913: 178, 914: -1, 915: -1, 916: -1, 917: -1, 918: -1, 919: -1, 920: -1, 921: -1, 922: -1, 923: -1, 924: 179, 925: -1, 926: -1, 927: -1, 928: -1, 929: -1, 930: -1, 931: -1, 932: 180, 933: 181, 934: 182, 935: -1, 936: -1, 937: 183, 938: -1, 939: -1, 940: -1, 941: -1, 942: -1, 943: 184, 944: -1, 945: 185, 946: -1, 947: 186, 948: -1, 949: -1, 950: -1, 951: 187, 952: -1, 953: -1, 954: 188, 955: -1, 956: 189, 957: 190, 958: -1, 959: 191, 960: -1, 961: -1, 962: -1, 963: -1, 964: -1, 965: -1, 966: -1, 967: -1, 968: -1, 969: -1, 970: -1, 971: 192, 972: 193, 973: -1, 974: -1, 975: -1, 976: -1, 977: -1, 978: -1, 979: -1, 980: 194, 981: 195, 982: -1, 983: -1, 984: 196, 985: -1, 986: 197, 987: 198, 988: 199, 989: -1, 990: -1, 991: -1, 992: -1, 993: -1, 994: -1, 995: -1, 996: -1, 997: -1, 998: -1, 999: -1}
        indices_in_1k = [k for k in thousand_k_to_200 if thousand_k_to_200[k] != -1]
    elif args.dataset == 'imagenet-r':
        all_wnids = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271', 'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525', 'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318', 'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007', 'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263', 'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074', 'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742', 'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110', 'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']
        imagenet_r_wnids = {'n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677'}
        indices_in_1k = [wnid in imagenet_r_wnids for wnid in all_wnids]
    elif args.dataset == 'imagenet-100-v2':
        # folder structure is different so order of labels is different for imagenet-100-v2
        with open('./misc/in100_class_to_idx.pkl', 'rb') as f:
            in100_class_to_idx = pickle.load(f)

        class_idx = json.load(open("./misc/imagenet_classes.json"))
        idx2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

        ds_class_to_idx_new = {}
        ds_class_to_idx = val_dataset.datasets[0].class_to_idx
        for k in ds_class_to_idx.keys():
            ds_class_to_idx_new[idx2label[int(k)]] = ds_class_to_idx[k]

        indices_in_100 = [ds_class_to_idx_new[k] for k in in100_class_to_idx.keys()]
        indices_in_1k = indices_in_100
    elif 'imagenet-100-' in args.dataset:
        with open('./misc/in100_class_to_idx.pkl', 'rb') as f:
            in100_class_to_idx = pickle.load(f)

        ds_class_to_idx = val_dataset.class_to_idx
        indices_in_100 = [k in ds_class_to_idx for k in in100_class_to_idx.keys()]
        indices_in_1k = indices_in_100
    else:
        if args.use_smaller_split:
            indices_in_1k = [True for _ in range(args.val_perc)]
        else:
            indices_in_1k = [True for _ in range(dataset_num_classes[args.dataset])]


    return train_loader, val_loader, test_loader, indices_in_1k

def classes_to_idxs(dataset):
    class_to_idx = {}
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'y'):
        targets = dataset.y
    elif hasattr(dataset, 'index'):
        targets = [dataset.index[i][0] for i in range(len(dataset))]
    elif hasattr(dataset, '_labels'):
        targets = dataset._labels

    print(f'Number of classes: {len(targets)}')
    for i in range(len(targets)):
        cur_class = targets[i]
        if cur_class not in class_to_idx:
            class_to_idx[cur_class] = [i,]
        else:
            class_to_idx[cur_class].append(i)

    return class_to_idx

class ModelDataset(Dataset):
    """
        Modified version of the usual ImageNet dataset: this dataset returns the label of the best
        model for each of the classes (baseline = 0, invariant = 1, equivariant = 2, none = 3 in that order)

        data_dir: path to data directory
        sampling_method: classwise or samplewise (whether to pick model based on sample of class)
    """
    def __init__(self, dataset_name, data_dir, sampling_method, transform=None, train=False):
        self.dataset_name = dataset_name
        self.dataset = ImageFolder(data_dir, transform=transform)
        self.sampling_method = sampling_method
        self.train = train

        if self.sampling_method == 'classwise':
            bei_filepath = './misc/bei_label_per_class.pth'
        elif self.sampling_method == 'classwise-soft':
            bei_filepath = './misc/bei_label_per_class_soft.pth'
        elif self.sampling_method == 'samplewise-single':
            if self.train:
                bei_filepath = './misc/bei_label_per_sample_single_train.pth'
            else:
                bei_filepath = './misc/bei_label_per_sample_single_val.pth'
        elif self.sampling_method == 'samplewise-single-soft':
            if self.train:
                bei_filepath = './misc/bei_label_per_sample_single_soft_train.pth'
            else:
                bei_filepath = './misc/bei_label_per_sample_single_soft_val.pth'
        elif self.sampling_method == 'samplewise-multi':
            if self.train:
                bei_filepath = './misc/bei_label_per_sample_multi_train.pth'
            else:
                bei_filepath = './misc/bei_label_per_sample_multi_val.pth'
        else:
            raise NotImplementedError(f'{self.sampling_method} sampling method not supported')

        if os.path.isfile(bei_filepath):
            self.bei_label = torch.load(bei_filepath, map_location='cpu')
        else:
            raise Exception('BEI label per class file not found')

        self.num_classes = len(torch.unique(self.bei_label))

        if dataset_name == 'imagenet-100':
            with open('./misc/in100_to_in1000_labels_dict.pkl', 'rb') as f:
                self.in100_to_in1000_label_dict = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.dataset_name == 'imagenet-100':
            label = self.in100_to_in1000_label_dict[label]

        if self.sampling_method == 'classwise':
            bei_label = self.bei_label[label].long()
        elif self.sampling_method == 'samplewise-single-soft':
            bei_label = self.bei_label[idx]
        else:
            bei_label = self.bei_label[idx].long()

        return img, bei_label

# Dataset
class Split_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label #, path

class TopKDataset(Dataset):
    def __init__(self, dataset, data_dir, topk_path, topk, transform=None):
        if dataset == 'imagenet':
            self.dataset = ImageFolder(data_dir, transform=transform)
        elif dataset == 'cifar100':
            self.dataset = CIFAR100(data_dir, transform=transform, train=True)
        self.topk = topk

        if os.path.isfile(topk_path):
            with open(topk_path, 'rb') as f:
                self.topk_dict = pickle.load(f)
        else:
            raise Exception('Pickle load failed')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (y1, y2), label = self.dataset[idx]

        topk_labels = self.topk_dict[idx].long()

        if len(topk_labels) < 10:
            pad = torch.empty(10-len(topk_labels)).fill_(-1.0).long()
            topk_labels = torch.cat([topk_labels, pad])

        return (y1, y2), (label, topk_labels[:self.topk])


class LogitsDataset(Dataset):
    def __init__(self, dataset, data_dir, logits_path, transform=None):
        if dataset == 'imagenet':
            self.dataset = ImageFolder(data_dir, transform=transform)
        elif dataset == 'cifar100':
            self.dataset = CIFAR100(data_dir, transform=transform, train=True)

        self.logits = np.memmap(logits_path, dtype='float32', mode='r', shape=(len(self.dataset),1000))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (y1, y2), label = self.dataset[idx]

        logits = torch.from_numpy(self.logits[idx])

        return (y1, y2), (label, logits)

class IdxDataset(Dataset):
    """Paired ImageFolder (for both ImageNet and StylizedImageNet, in that order)"""
    def __init__(self, dataset, root, transform=None, target_transform=None, is_valid_file=None, train=True):
        if dataset == 'cifar10':
            self.data = CIFAR10(root=root,
                                transform=transform,
                                train=train
                                )
        elif dataset == 'cifar100':
            self.data = CIFAR100(root=root,
                                 transform=transform,
                                 train=train)
        elif dataset == 'imagenet' or dataset == 'path':
            self.data = ImageFolder(root=root, transform=transform)
        else:
            raise ValueError(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]

        return x, y, idx

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, download=False, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class ObjectNetDataset(VisionDataset):
    """
    ObjectNet dataset.
    Args:
        root (string): Root directory where images are downloaded to. The images can be grouped in folders. (the folder structure will be ignored)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, 'transforms.ToTensor'
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        img_format (string): jpg
                             png - the original ObjectNet images are in png format
    """

    def __init__(self, root, transform=None, target_transform=None, transforms=None, img_format="jpg"):
        """Init ObjectNet pytorch dataloader."""
        super(ObjectNetDataset, self).__init__(root, transforms, transform, target_transform)

        self.loader = self.pil_loader
        self.img_format = img_format
        files = glob.glob(root+"/**/*."+img_format, recursive=True)
        self.pathDict = {}
        for f in files:
            self.pathDict[f.split("/")[-1]] = f
        self.imgs = list(self.pathDict.keys())

    def __getitem__(self, index):
        """
        Get an image and its label.
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the image file name
        """
        img, target = self.getImage(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def getImage(self, index):
        """
        Load the image and its label.
        Args:
            index (int): Index
        Return:
            tuple: Tuple (image, target). target is the image file name
        """
        img = self.loader(self.pathDict[self.imgs[index]])

        # crop out red border
        width, height = img.size
        cropArea = (2, 2, width-2, height-2)
        img = img.crop(cropArea)
        return (img, self.imgs[index])

    def __len__(self):
        """Get the number of ObjectNet images to load."""
        return len(self.imgs)

    def pil_loader(self, path):
        """Pil image loader."""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

class CIFAR10C(VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None):
        corruptions = load_txt('./misc/cifar10-c-corruptions.txt')
        assert name in corruptions

        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)

class Transform:
    def __init__(self, args):
        self.args = args
        if args.dataset == 'imagenet':

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            if args.weak_aug:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                saturation=0.2, hue=0.1)],
                        p=1.0
                    ),
                    transforms.RandomGrayscale(p=0.5),
                    GaussianBlur(p=1.0),
                    Solarization(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

                self.transform_prime = transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform_prime = transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                saturation=0.2, hue=0.1)],
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=0.1),
                    Solarization(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

            self.transform_rotation = transforms.Compose([
                transforms.RandomResizedCrop(args.downsize, scale=(args.scale[0], args.scale[1])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

        elif args.dataset == 'cifar100':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761])
            ])

            self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761])
            ])

            self.transform_rotation = self.transform_prime

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        y3 = self.transform_rotation(x)

        return y1, y2, y3

class SingleTransform:
    def __init__(self, args):
        self.args = args
        if args.dataset == 'imagenet':
            self.transform_rotation = transforms.Compose([
                transforms.RandomResizedCrop(args.downsize, scale=(args.scale[0], args.scale[1])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            raise "only for stylized imagenet"

    def __call__(self, x):
        y = self.transform_rotation(x)
        return y

class BiasedBatchSampler(BatchSampler):
    """Samples batches of data points that are biased/subset of a specified number of classes.

    Args:
        dataset: pytorch dataset
        batch_size (int): batch size.
        num_classes_per_batch (int): number of classes to sample from in a batch.
    Return
        yields the batch_ids/image_ids/image_indices

    """
    def __init__(self, dataset, batch_size, num_classes_per_batch, drop_last=False):
        self.dataset = dataset
        # self.sampler = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_classes_per_batch = num_classes_per_batch
        self.batch_count = math.ceil(len(self.dataset) / self.batch_size)

        # organize indices of each class into a dict
        self.class_to_idx = {}

        for i in range(len(self.dataset.targets)):
            cur_class = self.dataset.targets[i]
            if cur_class not in self.class_to_idx:
                self.class_to_idx[cur_class] = [i,]
            else:
                self.class_to_idx[cur_class].append(i)

    def __precompute_batches__(self):
        # shuffle indices of each class
        class_to_idx = self.class_to_idx.copy()

        for k in class_to_idx.keys():
            random.shuffle(class_to_idx[k])

        batch_idxs = []

        step = 0
        while len(class_to_idx.keys()) > 0:
            all_classes = list(class_to_idx.keys())

            # if leftover number of classes is smaller than # of classes to sample per batch, sample leftover only
            if self.num_classes_per_batch > len(all_classes):
                num_classes_per_batch = len(all_classes)
            else:
                num_classes_per_batch = self.num_classes_per_batch

            # sample classes for current batch and divide into equal number of partitions
            cur_batch_classes = np.random.choice(list(class_to_idx.keys()), num_classes_per_batch, replace=False)
            parts = [self.batch_size // num_classes_per_batch + (1 if x < self.batch_size % num_classes_per_batch else 0)  for x in range(num_classes_per_batch)]

            carryover = 0
            cur_batch_idxs = []

            # sampling equally from chosen number of classes
            for i in range(len(cur_batch_classes)):
                cur_class = cur_batch_classes[i]
                num_imgs = parts[i] + carryover

                cur_class_idxs = class_to_idx[cur_class]
                if len(cur_class_idxs) >= num_imgs:
                    cur_batch_idxs.extend(cur_class_idxs[:num_imgs])
                    if len(cur_class_idxs) == num_imgs:
                        class_to_idx.pop(cur_class)
                    else:
                        class_to_idx[cur_class] = cur_class_idxs[num_imgs:]
                    carryover = 0
                else:
                    cur_batch_idxs.extend(cur_class_idxs)
                    class_to_idx.pop(cur_class)
                    carryover = num_imgs - len(cur_class_idxs)


            # re-sampling from chosen classes once again if there is
            # carryover after first iteration of sampling
            for i in range(len(cur_batch_classes)):
                cur_class = cur_batch_classes[i]
                num_imgs = carryover

                if cur_class not in class_to_idx:
                  continue
                cur_class_idxs = class_to_idx[cur_class]
                if len(cur_class_idxs) >= num_imgs:
                    cur_batch_idxs.extend(cur_class_idxs[:num_imgs])
                    if len(cur_class_idxs) == num_imgs:
                        class_to_idx.pop(cur_class)
                    else:
                        class_to_idx[cur_class] = cur_class_idxs[num_imgs:]
                    carryover = 0
                else:
                    cur_batch_idxs.extend(cur_class_idxs)
                    class_to_idx.pop(cur_class)
                    carryover = num_imgs - len(cur_class_idxs)

                if carryover == 0:
                  break

            if (len(cur_batch_idxs)) != 128:
              print(len(cur_batch_idxs), step, carryover)


            random.shuffle(cur_batch_idxs)
            batch_idxs.extend(cur_batch_idxs)

            step += 1

        return batch_idxs

    def __iter__(self):
        batch_idxs = self.__precompute_batches__()

        for i in range(0, len(batch_idxs), self.batch_size):
            try:
                yield batch_idxs[i:i+self.batch_size]
            except IndexError:
                break

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)

# if __name__ == '__main__':
    # testds = Stylized_Dataset('/Users/charlotteloh/Documents/test.txt', version='inv')
    # testds[1]

    ### example of how to use BiasedBatchSampler

    # ds = datasets.CIFAR100('./', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    # sampler = BiasedBatchSampler(ds, 128, 20, drop_last=True)
    # dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler)
    # dl_iter = iter(dl)
    # x, y = next(dl_iter)
    # torch.unique(y).shape

    # dist_sampler = DistributedBatchSampler(sampler, num_replicas=2, rank=0)
