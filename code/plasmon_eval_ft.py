import torch
from torchvision import models
from torchvision import datasets, transforms
# from datasets import Split_Dataset
from tqdm import tqdm
import torch.nn.functional as F
# from torch.utils.data import Subset
import numpy as np
import torch.nn.functional as F
import os

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

test_dataset = datasets.ImageFolder('/data/cloh/imagenet_data/val', transform=val_transforms)

# val_dataset = Split_Dataset('/data/cloh/imagenet_data',  \
#                     f'./calib_splits/am_imagenet_5percent_val.txt',
#                     transform=val_transforms)

test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=False,
            num_workers=20, pin_memory=True,
        )
# val_loader = torch.utils.data.DataLoader(
#             val_dataset, batch_size=256, shuffle=False,
#             num_workers=20, pin_memory=True,
#         )

def load_1_model(ckpt_path, full_path=False):
    model1 = models.resnet50().cuda()
    if not full_path:
        sd = torch.load(f"./ensem_ssl_ft_ckpts/{ckpt_path}/checkpoint_best.pth", map_location="cpu")
    else:
        sd = torch.load(ckpt_path, map_location="cpu")
    ckpt = {k.replace("members.0.",""):v for k,v in sd['model'].items()}
    model1.load_state_dict(ckpt)
    print(f"loaded {ckpt_path}")
    model1.eval()
    return model1

def load_1_ts_model(ckpt_path, full_path=True):
    model1 = models.resnet50().cuda()
    model1.temp = torch.nn.Parameter(torch.ones(1) * 1.5)
    if not full_path:
        sd = torch.load(f"./ensem_ssl_ft_ckpts/{ckpt_path}/checkpoint_best.pth", map_location="cpu")
    else:
        sd = torch.load(ckpt_path, map_location="cpu")
    ckpt = {k.replace("members.0.",""):v for k,v in sd['model'].items()}
    model1.load_state_dict(ckpt)
    print(f"loaded {ckpt_path}")
    model1.eval()
    return model1

import torch.nn.functional as F

class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='sum', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p = F.log_softmax(p, dim=-1)
        q = F.log_softmax(q, dim=-1)

        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

class KLD(torch.nn.Module):
    def __init__(self):
        super(KLD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='sum', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p = F.log_softmax(p, dim=-1)
        q = F.log_softmax(q, dim=-1)
        return self.kl(p,q)

# kl_div = KLD()
# js_div = JSD()

class _ECELoss(torch.nn.Module):

    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
#         softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

nll_criterion = torch.nn.CrossEntropyLoss().cuda()
ece_criterion = _ECELoss().cuda()

def compute_pair_consensus(pair_preds):
    agree = (pair_preds[0] == pair_preds[1])
    agree_correct = agree & (pair_preds[0] == target)
    agree_wrong = agree & (pair_preds[0] != target)
    disagree = (pair_preds[0] != pair_preds[1])
    disagree_both_wrong = disagree & (pair_preds[0] != target) & (pair_preds[1] != target)
    disagree_one_correct = disagree & (pair_preds[0] != target) & (pair_preds[1] == target)
    disagree_one_correct2 = disagree & (pair_preds[1] != target) & (pair_preds[0] == target)
    return agree.sum(), disagree.sum(), agree_correct.sum(), agree_wrong.sum(), disagree_both_wrong.sum(), disagree_one_correct.sum()+disagree_one_correct2.sum()

def get_div_metrics(output1,output2,output3):
    preds = torch.stack([output1,output2,output3])
    avg_std_logits = torch.std(preds, dim=0).mean(dim=-1).mean() # std over members, mean over classes, sum over samples (mean taken later))
    avg_std = torch.std(preds.softmax(-1), dim=0).mean(dim=-1).mean() # std over members, mean over classes, sum over samples (mean taken later))
    _, all_preds = preds.max(-1)
    ag_p, dag_p, ag_c_p, ag_w_p, dag_w_p, dag_c_p = 0, 0, 0, 0, 0, 0
    kld = 0.
    for p in pairs:
        ag, dag, ag_c, ag_w, dag_w, dag_c = compute_pair_consensus(all_preds[p,:])
        ag_p += ag
        dag_p += dag
        ag_c_p += ag_c
        ag_w_p += ag_w
        dag_c_p += dag_c
        dag_w_p += dag_w
        kld += kl_div(preds[p[0]], preds[p[1]])

    ag_sum = ag_p/len(pairs)
    dag_sum = dag_p/len(pairs)
    ag_c_sum = ag_c_p/len(pairs)
    ag_w_sum = ag_w_p/len(pairs)
    dag_c_sum = dag_c_p/len(pairs)
    dag_w_sum = dag_w_p/len(pairs)
    kld_sum = kld/len(pairs)
    return ag_sum/len(output1), dag_sum/len(output1), ag_c_sum/len(output1), ag_w_sum/len(output1), dag_c_sum/len(output1), dag_w_sum/len(output1), kld_sum/len(output1)


class _ECELoss(torch.nn.Module):

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
#         softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def rollout_loader(model, loader):
    targets = []
    outputs = []
    for it, (img, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        img = img.cuda(non_blocking=True)
        with torch.no_grad():
            output1 = model(img)
#             ece_1 = ece_criterion(output1.softmax(-1), target)
            targets.append(target)
            outputs.append(output1)
    return torch.cat(outputs), torch.cat(targets)

import torch.nn.functional as F
import inspect
# from netcal.metrics import ECE

cecriterion = torch.nn.CrossEntropyLoss().cuda()
nll_criterion = torch.nn.CrossEntropyLoss().cuda()
ece_criterion = _ECELoss().cuda()
# ece_netcal = ECE(15)

test_loader2 = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=False,
            num_workers=20, pin_memory=True,
        )

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def get_metrics(outs, tars, names, printing=True):
    for out, tar,name in zip(outs,tars,names):
        # ece1 = ece_netcal.measure(out.softmax(-1).cpu().numpy(), tar.cpu().numpy())
        ece2 = ece_criterion(out.softmax(-1), tar)
        loss = F.nll_loss(torch.log(out.softmax(-1)), tar)
        loss2 = cecriterion(out, tar)
        _, pred = out.max(-1)
        acc = ((pred == tar).sum()) / len(tar)
        if printing:
            print(name)
            print("NLL:", loss.item(), loss2.item())
            print("ECE:", ece2.item())
            print("Acc:", acc.item())
    return loss.item(), ece2.item(), acc.item()

def get_metrics_softmax(outs, tars, names, printing=True):
    for out, tar, name in zip(outs,tars,names):
        # ece1 = ece_netcal.measure(out.cpu().numpy(), tar.cpu().numpy())
        ece2 = ece_criterion(out, tar)
        loss = F.nll_loss(torch.log(out), tar)
        _, pred = out.max(-1)
        acc = ((pred == tar).sum()) / len(tar)
        if printing:
            print(name)
            print("NLL:", loss.item())
            print("ECE:", ece2.item())
            print("Acc:", acc.item())
    return loss.item(), ece2.item(), acc.item()

os.makedirs("./logits_files", exist_ok=True)

# mod1 = load_1_model("./ensem_ssl_ft_ckpts/inv24_FT_checkpoint_best.pth",full_path=True)
# out, tar = rollout_loader(mod1, test_loader2)
# get_metrics([out],[tar],['inv24'])
# torch.save(out.cpu(), "./logits_files/inv24.pth")
# torch.save(tar.cpu(), "./logits_files/inv24_tar.pth")
#
# mod1 = load_1_model("./ensem_ssl_ft_ckpts/roteq-IN1k-e800-seed10-ft-cos-lr0.003-bs258_checkpoint_best.pth",full_path=True)
# out, tar = rollout_loader(mod1, test_loader2)
# get_metrics([out],[tar],['eq10'])
# torch.save(out.cpu(), "./logits_files/eq10.pth")
# torch.save(tar.cpu(), "./logits_files/eq10_tar.pth")
#
# mod1 = load_1_model("./ensem_ssl_ft_ckpts/roteq-IN1k-e800-seed78-ft-cos-lr0.003-bs258_checkpoint_best.pth",full_path=True)
# out, tar = rollout_loader(mod1, test_loader2)
# get_metrics([out],[tar],['eq78'])
# torch.save(out.cpu(), "./logits_files/eq78.pth")
# torch.save(tar.cpu(), "./logits_files/eq78_tar.pth")

# mod1 = load_1_model("./ensem_ssl_ft_ckpts/ft_baseR_cos_lr0.003_bs256_ftseed1/checkpoint_best.pth",full_path=True)
# out, tar = rollout_loader(mod1, test_loader2)
# get_metrics([out],[tar],['baseR'])
# torch.save(out.cpu(), "./logits_files/baseR_fseed1.pth")
# torch.save(tar.cpu(), "./logits_files/baseR_fseed1.pth")
#
# mod1 = load_1_model("./ensem_ssl_ft_ckpts/ft_baseR_cos_lr0.003_bs256_ftseed2/checkpoint_best.pth",full_path=True)
# out, tar = rollout_loader(mod1, test_loader2)
# get_metrics([out],[tar],['baseR'])
# torch.save(out.cpu(), "./logits_files/baseR_fseed2.pth")
#
# mod1 = load_1_model("./ensem_ssl_ft_ckpts/ft_baseR_cos_lr0.003_bs256_ftseed3/checkpoint_best.pth",full_path=True)
# out, tar = rollout_loader(mod1, test_loader2)
# get_metrics([out],[tar],['baseR'])
# torch.save(out.cpu(), "./logits_files/baseR_fseed3.pth")

mod1 = load_1_model("./ensem_ssl_ft_ckpts/ft_eqR_cos_lr0.003_bs256_ftseed1/checkpoint_best.pth",full_path=True)
out, tar = rollout_loader(mod1, test_loader2)
get_metrics([out],[tar],['eR'])
torch.save(out.cpu(), "./logits_files/eqR_fseed1.pth")
torch.save(tar.cpu(), "./logits_files/eqR_fseed1.pth")

mod1 = load_1_model("./ensem_ssl_ft_ckpts/ft_eqR_cos_lr0.003_bs256_ftseed2/checkpoint_best.pth",full_path=True)
out, tar = rollout_loader(mod1, test_loader2)
get_metrics([out],[tar],['eR2'])
torch.save(out.cpu(), "./logits_files/eqR_fseed2.pth")

mod1 = load_1_model("./ensem_ssl_ft_ckpts/ft_eqR_cos_lr0.003_bs256_ftseed3/checkpoint_best.pth",full_path=True)
out, tar = rollout_loader(mod1, test_loader2)
get_metrics([out],[tar],['eR3'])
torch.save(out.cpu(), "./logits_files/eqR_fseed3.pth")
# torch.save(eq69_out.cpu(), "./logits_files/eq69.pth")
# torch.save(eq69_tar.cpu(), "./logits_files/targets.pth")

#
# eq69 = load_1_model("ft_eq69_cos_lr0.003_bs256")
# eq69_out, eq69_tar = rollout_loader(eq69, test_loader2)
# torch.save(eq69_out.cpu(), "./logits_files/eq69.pth")
# torch.save(eq69_tar.cpu(), "./logits_files/targets.pth")
#
# eq42 = load_1_model("ft_eq42_cos_lr0.003_bs256")
# eq42_out, eq42_tar = rollout_loader(eq42, test_loader2)
# torch.save(eq42_out.cpu(), "./logits_files/eq42.pth")
#
# eq31 = load_1_model("ft_eq31_cos_lr0.003_bs256")
# eq31_out, eq31_tar = rollout_loader(eq31, test_loader2)
# torch.save(eq31_out.cpu(), "./logits_files/eq31.pth")
#
# eq24 = load_1_model("ft_eq24_cos_lr0.003_bs256")
# eq24_out, eq24_tar = rollout_loader(eq24, test_loader2)
# torch.save(eq24_out.cpu(), "./logits_files/eq24.pth")
#
# baseR = load_1_model("ft_baseR_cos_lr0.003_bs256")
# baseR_out, baseR_tar = rollout_loader(baseR, test_loader2)
# torch.save(baseR_out.cpu(), "./logits_files/baseR.pth")
#
# base31 = load_1_model("ft_base31_cos_lr0.003_bs256")
# base31_out, base31_tar = rollout_loader(base31, test_loader2)
# torch.save(base31_out.cpu(), "./logits_files/base31.pth")
#
# base24 = load_1_model("ft_base24_cos_lr0.003_bs256")
# base24_out, base24_tar = rollout_loader(base24, test_loader2)
# torch.save(base24_out.cpu(), "./logits_files/base24.pth")
#
# base69 = load_1_model("ft_base69_cos_lr0.003_bs256")
# base69_out, base69_tar = rollout_loader(base69, test_loader2)
# torch.save(base69_out.cpu(), "./logits_files/base69.pth")
#
# inv = load_1_model("ft_inv_cos_lr0.003_bs256")
# inv_out, inv_tar = rollout_loader(inv, test_loader2)
# torch.save(inv_out.cpu(), "./logits_files/inv.pth")
#
# inv24 = load_1_model("ft_inv24_cos_lr0.004_bs256")
# inv24_out, inv24_tar = rollout_loader(inv24, test_loader2)
# torch.save(inv24_out.cpu(), "./logits_files/inv24.pth")
#
# inv31 = load_1_model("ft_inv31_cos_lr0.004_bs256")
# inv31_out, inv31_tar = rollout_loader(inv31, test_loader2)
# torch.save(inv31_out.cpu(), "./logits_files/inv31.pth")
#
# inv69 = load_1_model("ft_inv69_cos_lr0.004_bs256")
# inv69_out, inv69_tar = rollout_loader(inv69, test_loader2)
# torch.save(inv69_out.cpu(), "./logits_files/inv69.pth")
#
# eq54 = load_1_model("ft_eq54_cos_lr0.003_bs256")
# eq54_out, eq54_tar = rollout_loader(eq54, test_loader2)
# torch.save(eq54_out.cpu(), "./logits_files/eq54.pth")
#
# eq96 = load_1_model("roteq-IN1k-e800-seed96-ft-cos-lr0.003-bs258.pth", full_path=True)
# eq96_out, eq96_tar = rollout_loader(eq96, test_loader2)
# torch.save(eq96_out.cpu(), "./logits_files/eq96.pth")
#

# all_eq = [eq69_out, eq42_out, eq31_out, eq24_out, eq96_out]
# tars = [eq69_tar, eq42_tar, eq31_tar, eq24_tar, eq96_tar]
# names = ['eq69', 'eq42', 'eq31', 'eq24', 'eq96']
# get_metrics(all_eq,tars,names)
#
# all_base = [baseR_out, base31_out, base24_out, base69_out]
# tars = [baseR_tar, base31_tar, base24_tar, base69_tar]
# names = ['baseR', 'base31', 'base24', 'base69']
# get_metrics(all_base,tars, names)
#
