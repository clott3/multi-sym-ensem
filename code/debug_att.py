import torch
from torchvision import models
from torchvision import datasets, transforms
from datasets import Split_Dataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

test_dataset = datasets.ImageFolder('/gpfs/u/locker/200/CADS/datasets/ImageNet/val', transform=val_transforms)

val_dataset = Split_Dataset('/gpfs/u/locker/200/CADS/datasets/ImageNet',  \
                    f'./calib_splits/am_imagenet_5percent_val.txt',
                    transform=val_transforms)

test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=True,
            num_workers=20, pin_memory=True,
        )
val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=20, pin_memory=True,
        )

indices = np.arange(len(test_dataset))
test_dataset = Subset(test_dataset, list(indices))

subset_test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=False,
            num_workers=20, pin_memory=True,
        )

print(subset_test_loader.dataset.dataset)
