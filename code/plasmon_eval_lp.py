import torch
import matplotlib.pyplot as plt
import os
os.getcwd()
import numpy as np
from datasets import Split_Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
val_transforms = transforms.Compose([
        transforms.Resize(224+32),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

val_ds = ImageFolder('/data/cloh/imagenet_data/val', transform=val_transforms)

num_classes = 200

test_dataset = Split_Dataset('/data/cloh/imagenet_data',  \
                            f'./calib_splits/am_IN200_val.txt',
                            transform=val_transforms)
print(f'Loading ./calib_splits/am_IN{num_classes}_val.txt')

# test_loader = torch.utils.data.DataLoader(
#             val_ds, batch_size=256, shuffle=False,
#             num_workers=20, pin_memory=True,
#         )
test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=False,
            num_workers=20, pin_memory=True,
        )

from eval_metrics import load_1_model, get_metrics, rollout_loader


eq69 = load_1_model(f"/home/cloh/ensem_ssl_lp_ckpts/roteq-IN{num_classes}-e400-seed69-lp-cos-lr0.3-bs258-checkpoint_best.pth", full_path=True, num_classes=num_classes)
eq69_out, same_tar = rollout_loader(eq69, test_loader)

base69 = load_1_model(f"/home/cloh/ensem_ssl_lp_ckpts/IN{num_classes}_lp_ckpts/base-IN{num_classes}-e400-seed69-lp-cos-lr0.3-bs258-checkpoint_best.pth", full_path=True, num_classes=num_classes)
base69_out, _ = rollout_loader(base69, test_loader)

inv69 = load_1_model(f"/home/cloh/ensem_ssl_lp_ckpts/IN{num_classes}_lp_ckpts/rotinv-IN{num_classes}-e400-seed69-lp-cos-lr0.3-bs258-checkpoint_best.pth", full_path=True, num_classes=num_classes)
inv69_out, _ = rollout_loader(inv69, test_loader)
