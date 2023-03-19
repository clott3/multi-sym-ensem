import torch
from torchvision import models
from torchvision import datasets, transforms
from datasets import Split_Dataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
import torch.nn.functional as F

import torch.nn.functional as F
import inspect
from netcal.metrics import ECE

cecriterion = torch.nn.CrossEntropyLoss().cuda()
nll_criterion = torch.nn.CrossEntropyLoss().cuda()
# ece_criterion = _ECELoss().cuda()
ece_netcal = ECE(15)


def load_1_model(ckpt_path, full_path=False, num_classes=1000):
    model1 = models.resnet50(num_classes=num_classes).cuda()
    if not full_path:
        sd = torch.load(f"./dist_models/{ckpt_path}/checkpoint_best.pth", map_location="cpu")
    else:
        sd = torch.load(ckpt_path, map_location="cpu")
    ckpt = {k.replace("members.0.",""):v for k,v in sd['model'].items()}
    model1.load_state_dict(ckpt)
    print(f"loaded {ckpt_path}")
    model1.eval()
    return model1

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

def get_metrics(outs, tars, names, printing=True, input_softmax=False, num_classes=1000):

    for out, tar,name in zip(outs,tars,names):
        out = out.cpu()
        tar = tar.cpu()
        correct_per_class = torch.zeros(num_classes).to(tar.device)
        total_per_class = torch.zeros(num_classes).to(tar.device)

        if not input_softmax:
            out = out.softmax(-1)
        ece1 = ece_netcal.measure(out.numpy(), tar.numpy())
#         ece2 = ece_criterion(out, tar)
        loss = F.nll_loss(torch.log(out), tar)
        _, pred = out.max(-1)
        correct_vec = (pred == tar)
        ind_per_class = (tar.unsqueeze(1) == torch.arange(num_classes).to(tar.device)) # indicator variable for each class
        correct_per_class = (correct_vec.unsqueeze(1) * ind_per_class).sum(0)
        total_per_class = ind_per_class.sum(0)

        acc = (correct_vec.sum()) / len(tar)
        acc_per_class = correct_per_class / total_per_class
        if printing:
            print(name)
            print(f"NLL: {loss.item()} | ECE: {ece1}")
            print("Acc:", acc.item())
    return loss.item(), ece1, acc.item(), acc_per_class

class KLD(torch.nn.Module):
    def __init__(self):
        super(KLD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='sum', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p = F.log_softmax(p, dim=-1)
        q = F.log_softmax(q, dim=-1)
        return self.kl(p,q)

        # kl_div = KLD()

def compute_pair_consensus(pair_preds, target):
    agree = (pair_preds[0] == pair_preds[1])
    agree_correct = agree & (pair_preds[0] == target)
    agree_wrong = agree & (pair_preds[0] != target)
    disagree = (pair_preds[0] != pair_preds[1])
    disagree_both_wrong = disagree & (pair_preds[0] != target) & (pair_preds[1] != target)
    disagree_one_correct = disagree & (pair_preds[0] != target) & (pair_preds[1] == target)
    disagree_one_correct2 = disagree & (pair_preds[1] != target) & (pair_preds[0] == target)
    return agree.sum(), disagree.sum(), agree_correct.sum(), agree_wrong.sum(), disagree_both_wrong.sum(), disagree_one_correct.sum()+disagree_one_correct2.sum()

def get_div_metrics(output1,output2,output3,target):
    preds = torch.stack([output1,output2,output3])
    avg_std_logits = torch.std(preds, dim=0).mean(dim=-1).mean() # std over members, mean over classes, sum over samples (mean taken later))
    avg_std = torch.std(preds.softmax(-1), dim=0).mean(dim=-1).mean() # std over members, mean over classes, sum over samples (mean taken later))
    _, all_preds = preds.max(-1)
    ag_p, dag_p, ag_c_p, ag_w_p, dag_w_p, dag_c_p = 0, 0, 0, 0, 0, 0
    kld = 0.
    pairs = ([0,1], [0,2], [1,2])
    for p in pairs:
        ag, dag, ag_c, ag_w, dag_w, dag_c = compute_pair_consensus(all_preds[p,:], target)
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
    print(f"Diversity agree: {ag_sum/len(output1)} | disagree: {dag_sum/len(output1)}")
#     print(f"Ensemble Variance Logits: {avg_std_logits} ")
#     print(f"Ensemble Variance: {avg_std}")
#     print(f"KL div: {kld_sum/len(output1)}")
    return ag_sum/len(output1), dag_sum/len(output1), kld_sum/len(output1), avg_std_logits, avg_std

def get_classwise(acc_base, acc_rotinv, acc_roteq, num_classes=1000):
    print("use order B, I, E")
    y = torch.stack([v for v in [acc_base, acc_rotinv, acc_roteq]], dim=-1)

    fac = num_classes/100
    # all 3 models equally good
    best_base_inv_eq = (y[:,0] == y[:,1]) & (y[:,1] == y[:,2])
    # 2 models equally good and is better
    best_base_inv = (y[:,0] == y[:,1]) & (y[:,0] > y[:,2])
    best_base_eq = (y[:,0] == y[:,2]) & (y[:,0] > y[:,1])
    best_inv_eq = (y[:,1] == y[:,2]) & (y[:,1] > y[:,0])
    # 2 models equally good and is worse
    worse_base_inv = (y[:,0] == y[:,1]) & (y[:,0] < y[:,2]) # best eq
    worse_base_eq = (y[:,0] == y[:,2]) & (y[:,0] < y[:,1]) # best inv
    worse_inv_eq = (y[:,1] == y[:,2]) & (y[:,1] < y[:,0]) # best base
    all_diff = (y[:,0] != y[:,1]) & (y[:,1] != y[:,2]) & (y[:,0] != y[:,2])
    print(f"all equal best: {(best_base_inv_eq.sum())/fac:.1f}%")
    print(f"B,I equal best: {(best_base_inv.sum())/fac:.1f}%")
    print(f"B,E equal best: {(best_base_eq.sum())/fac:.1f}%")
    print(f"I,E equal best: {(best_inv_eq.sum())/fac:.1f}%")
    # print(f"all diff: {all_diff.sum()}")
    total = best_base_inv_eq.sum() + best_base_inv.sum() + best_base_eq.sum() + best_inv_eq.sum() + all_diff.sum() + worse_inv_eq.sum() + worse_base_eq.sum() + worse_base_inv.sum()
    assert total == num_classes

    # for all diff
    best_base = (y[:,0] > y[:,1]) & (y[:,0] > y[:,2]) & all_diff
    best_inv = (y[:,1] > y[:,0]) & (y[:,1] > y[:,2]) & all_diff
    best_eq = (y[:,2] > y[:,0]) & (y[:,2] > y[:,1]) & all_diff
    total_unique = best_base.sum()+best_inv.sum()+best_eq.sum()
    assert total_unique == all_diff.sum()

    # single model uniquely best
    b_uniq = best_base | worse_inv_eq
    i_uniq = best_inv | worse_base_eq
    e_uniq = best_eq | worse_base_inv
    print(f"B uniquely best: {b_uniq.sum()/fac:.1f}%")
    print(f"I uniquely best: {(best_inv.sum() + worse_base_eq.sum())/fac:.1f}%")
    print(f"E uniquely best: {(best_eq.sum() + worse_base_inv.sum())/fac:.1f}%")

    B_good = b_uniq | best_base_inv_eq | best_base_inv | best_base_eq
    I_good = i_uniq | best_base_inv_eq | best_base_inv | best_inv_eq
    E_good = e_uniq | best_base_inv_eq | best_base_eq | best_inv_eq
