from re import M
import torch
import torch.nn as nn
from torchvision import models

import collections

from utils import consume_prefix_in_state_dict_if_present
from gate_resnet import resnet0, resnet50
from vision_transformers import vit_tiny, vit_small, vit_base
import sys
import cifar_resnet

class EnsembleSSL(nn.Module):
    def __init__(self, arch, num_ensem=1, num_classes=1000, eval_mode='freeze'):
        super().__init__()
        print("initialized EnsembleSSL")
        self.num_ensem = num_ensem
        self.num_classes = num_classes
        model_fn = models_dict[arch]
        self.members = torch.nn.ModuleList([model_fn(num_classes=self.num_classes) for _ in range(num_ensem)])
        self.set_eval_mode(eval_mode)

    def load_sep_weights(self, weights_path_list):
        print("loading sep weights")
        for m in range(self.num_ensem):
            weights = weights_path_list[m]
            state_dict = torch.load(weights, map_location='cpu')

            cur_mem = self.members[m]
            if 'model' in state_dict:
                if 'members.0.fc.weight' in state_dict['model']:
                    # for LP on imagenet-100 using imagenet ckpt (i.e. different num classes)
                    if state_dict['model']['members.0.fc.weight'].shape[0] != self.num_classes:
                        print(f"model weights dim: {state_dict['model']['members.0.fc.weight'].shape[0]}, num classes: {self.num_classes}")
                        new_state_dict = {k:v for k,v in state_dict['model'].items() if 'fc' not in k}
                    else:
                        new_state_dict = state_dict['model']
                    consume_prefix_in_state_dict_if_present(new_state_dict, 'members.0.') # this is assuming only 1 member was trained at a time
                    missing_keys, unexpected_keys = cur_mem.load_state_dict(new_state_dict, strict=False)
                    print('missing keys', missing_keys)
                    print('unexpected keys', unexpected_keys)

                else:
                    consume_prefix_in_state_dict_if_present(state_dict['model'], 'members.0.') # this is assuming only 1 member was trained at a time
                    consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.backbone.')
                    missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict['model'], strict=False)
                    print('missing keys', missing_keys)
                    print('unexpected keys', unexpected_keys)
                    print("===> Loaded backbone state dict from ", weights)

            elif 'backbone' in state_dict:
                missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["backbone"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)
            elif 'state_dict' in state_dict:
                consume_prefix_in_state_dict_if_present(state_dict['state_dict'], 'encoder.')
                missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["state_dict"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)
            else:
                print(state_dict.keys())
                raise "state dict not found!"

            if 'log_reg_weight' in state_dict and 'log_reg_bias' in state_dict:
                cur_mem.fc.weight = torch.from_numpy(state_dict['log_reg_weight'])
                cur_mem.fc.bias = torch.from_numpy(state_dict['log_reg_bias'])


    def load_weights(self, weights_path_list, convert_from_single=False):
        # convert_from_single: whether to convert the weights from 1 model (MultiBackbone)
        # making sure that the number of pretrained weights & ensem member size is equal
        print("loading weights")
        if not convert_from_single:
            assert len(self.members) == len(weights_path_list)
        else:
            assert len(weights_path_list) == 1
            ensem_state_dict = torch.load(weights_path_list[0], map_location='cpu')
            # ensem_state_dict = ensem_state_dict['enc_state_dict']
            state_dicts = convert_weights_from_single_backbone(ensem_state_dict, self.num_ensem)

        for m in range(self.num_ensem):
            cur_mem = self.members[m]

            if convert_from_single:
                weights = ''
                state_dict = state_dicts[m]
            else:
                weights = weights_path_list[m]
                state_dict = torch.load(weights, map_location='cpu')

            if 'epoch' in str(weights):
                consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.backbone.')
                cur_mem.load_state_dict(state_dict['model'], strict=False)
            elif 'simsiam' in str(weights):
                consume_prefix_in_state_dict_if_present(state_dict['state_dict'], 'module.backbone.')
                consume_prefix_in_state_dict_if_present(state_dict['state_dict'], 'backbone.')
                missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["state_dict"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)
            else:
                if 'model' in state_dict:
                    missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["model"], strict=False)
                elif 'backbone' in state_dict:
                    missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["backbone"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)
                # assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []

            if self.eval_mode in {'linear_probe', 'finetune'}:
                cur_mem.fc.weight.data.normal_(mean=0.0, std=0.01)
                cur_mem.fc.bias.data.zero_()

    def set_eval_mode(self, mode='freeze'):
        self.eval_mode = mode
        if self.eval_mode == 'freeze':
            for cur_mem in self.members:
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(False)
        elif self.eval_mode == 'linear_probe':
            for cur_mem in self.members:
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(True)
        elif self.eval_mode == 'finetune':
            for cur_mem in self.members:
                cur_mem.requires_grad_(True)
                cur_mem.fc.requires_grad_(True)
        elif self.eval_mode == 'log_reg':
            for cur_mem in self.members:
                cur_mem.fc = nn.Identity()
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(False)
        elif self.eval_mode == 'extract_features':
            for cur_mem in self.members:
                # replacing all members' fc layers with identity to extract features
                cur_mem.fc = nn.Identity()

                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(False)
        else:
            raise NotImplementedError(f'Evaluation mode {mode} not implemented')


    # forward that feeds inputs to the ensemble and returns stacked outputs
    def forward(self, x, gate_cond=None):
        outputs = []

        for cur_mem in self.members:
            output = cur_mem(x)
            outputs.append(output)

        return torch.stack(outputs)

    # forward that feeds inputs to the ensemble and returns only the averaged outputs from the ensemble
    def forward_ensem(self, x):
        outputs = self.forward(x)

        return outputs.softmax(dim=-1).mean(dim=0)

class TS_EnsembleSSL(nn.Module):
    def __init__(self, arch, num_ensem=1, num_classes=1000, eval_mode='freeze'):
        super().__init__()
        print("initialized TS_EnsembleSSL")

        self.num_ensem = num_ensem
        self.num_classes = num_classes
        model_fn = models_dict[arch]
        self.members = torch.nn.ModuleList([model_fn(num_classes=self.num_classes) for _ in range(num_ensem)])
        self.set_eval_mode(eval_mode)
        if eval_mode == 'temp_scale':
            self.temp = nn.Parameter(torch.ones(1) * 1.5)
        elif eval_mode == 'temp_scale_frozen':
            self.temps = torch.nn.ModuleList([nn.Parameter(torch.ones(1) * 1.5) for _ in range(num_ensem)])
        else:
            self.temps = None

    def load_sep_weights(self, weights_path_list):
        for m in range(self.num_ensem):
            weights = weights_path_list[m]
            state_dict = torch.load(weights, map_location='cpu')

            cur_mem = self.members[m]
            if 'model' in state_dict:
                consume_prefix_in_state_dict_if_present(state_dict['model'], 'members.0.') # this is assuming only 1 member was trained at a time
                cur_mem.load_state_dict(state_dict['model'], strict=True)
            elif 'backbone' in state_dict:
                missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["backbone"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)

            if 'log_reg_weight' in state_dict and 'log_reg_bias' in state_dict:
                cur_mem.fc.weight = torch.from_numpy(state_dict['log_reg_weight'])
                cur_mem.fc.bias = torch.from_numpy(state_dict['log_reg_bias'])


    def load_weights(self, weights_path_list, convert_from_single=False):
        # convert_from_single: whether to convert the weights from 1 model (MultiBackbone)
        # making sure that the number of pretrained weights & ensem member size is equal
        if not convert_from_single:
            assert len(self.members) == len(weights_path_list)
        else:
            assert len(weights_path_list) == 1
            ensem_state_dict = torch.load(weights_path_list[0], map_location='cpu')
            # ensem_state_dict = ensem_state_dict['enc_state_dict']
            state_dicts = convert_weights_from_single_backbone(ensem_state_dict, self.num_ensem)

        for m in range(self.num_ensem):
            cur_mem = self.members[m]

            if convert_from_single:
                weights = ''
                state_dict = state_dicts[m]
            else:
                weights = weights_path_list[m]
                state_dict = torch.load(weights, map_location='cpu')

            if 'epoch' in str(weights):
                consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.backbone.')
                cur_mem.load_state_dict(state_dict['model'], strict=False)
            elif 'simsiam' in str(weights):
                consume_prefix_in_state_dict_if_present(state_dict['state_dict'], 'module.backbone.')
                consume_prefix_in_state_dict_if_present(state_dict['state_dict'], 'backbone.')
                missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["state_dict"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)
            else:
                missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["backbone"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)
                # assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []

            if self.eval_mode in {'linear_probe', 'finetune'}:
                cur_mem.fc.weight.data.normal_(mean=0.0, std=0.01)
                cur_mem.fc.bias.data.zero_()

    def set_eval_mode(self, mode='freeze'):
        self.eval_mode = mode
        if self.eval_mode == 'freeze' or self.eval_mode == 'temp_scale':
            for cur_mem in self.members:
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(False)
        elif self.eval_mode == 'linear_probe':
            for cur_mem in self.members:
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(True)
        elif self.eval_mode == 'finetune':
            for cur_mem in self.members:
                cur_mem.requires_grad_(True)
                cur_mem.fc.requires_grad_(True)
        elif self.eval_mode == 'log_reg':
            for cur_mem in self.members:
                cur_mem.fc = nn.Identity()
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(False)
        else:
            raise NotImplementedError(f'Evaluation mode {mode} not implemented')

    # def temperature_scale(self, logits):
    #     """
    #     Perform temperature scaling on logits (list of length == num_ensem)
    #     """
    #     outputs = []
    #     # Expand temperature to match the size of logits
    #     for i in range(len(logits)):
    #         temperature = self.temps[i].unsqueeze(1).expand(logits[i].size(0), logits[i].size(1))
    #         outputs.append(logits[i]/temperature)
    #     return outputs

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temp.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits/temperature

    # forward that feeds inputs to the ensemble and returns stacked outputs
    def forward(self, x, gate_cond=None):
        outputs = []
        for cur_mem in self.members:
            output = cur_mem(x)
            outputs.append(output)
        # if self.temps is not None:
            # outputs = self.temperature_scale(outputs)

        return torch.stack(outputs)

    # forward that feeds inputs to the ensemble and returns only the averaged outputs from the ensemble
    def forward_ensem(self, x):
        outputs = self.forward(x)
        if self.temps is not None:
            raise NotImplementedError(f'temperature scaling not implemented')
        return outputs.softmax(dim=-1).mean(dim=0)

class MLP(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        layers = []
        layers.append(nn.Linear(2048*3, hidden_dim[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        outputs = self.layers(x)

        return outputs

def convert_weights_from_single_backbone(state_dict, num_enc):
    enc_keys = sorted(state_dict['enc_state_dict'].keys())
    clf_keys = sorted(state_dict['class_state_dict'].keys())
    new_state_dicts = [collections.defaultdict(dict) for _ in range(num_enc)]

    for i in range(num_enc):
        prefix = f'encoders.member{i}.'
        for key in enc_keys:
            if key.startswith(prefix):
                newkey = key[len(prefix):]
                new_state_dicts[i]['backbone'][newkey] = state_dict['enc_state_dict'].pop(key)

        prefix = f'branches.member{i}.'
        for key in clf_keys:
            if key.startswith(prefix):
                newkey = 'fc.' + key[len(prefix):]
                new_state_dicts[i]['backbone'][newkey] = state_dict['class_state_dict'].pop(key)

    return new_state_dicts



class GatedEnsembleSSL(nn.Module):
    def __init__(self, arch, num_ensem=1, num_classes=1000, eval_mode='freeze', gate_arch='mlp', smallmlp_hd=512, vit_patch_size=2048):
        super().__init__()
        print("initialized GatedEnsembleSSL")

        self.num_ensem = num_ensem
        self.num_classes = num_classes
        # model_fn = models_dict[arch]
        model_fn = resnet50 # need to use this for return_rep
        self.members = torch.nn.ModuleList([model_fn(num_classes=self.num_classes) for _ in range(num_ensem)])
        print(f"using {gate_arch} gate network")

        if gate_arch == 'mlp_selector':
            self.gate = torch.nn.Sequential(torch.nn.Linear(2048*num_ensem, 1024),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    torch.nn.Linear(1024, 512),
                                                    torch.nn.ReLU(inplace=True),  # second layer
                                                    torch.nn.Linear(512, num_ensem)
                                                    )  # output layer
        elif gate_arch == 'rn18_selector':
            self.gate = models_dict['resnet18'](num_classes=num_ensem)

        elif gate_arch == 'mlp':
            self.gate = torch.nn.Sequential(torch.nn.Linear(2048*num_ensem, 1024),
                                                    torch.nn.LayerNorm(1024),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    torch.nn.Linear(1024, 512),
                                                    torch.nn.LayerNorm(512),
                                                    # torch.nn.ReLU(inplace=True),  # second layer
                                                    torch.nn.Linear(512, num_ensem)
                                                    )  # output layer
        elif gate_arch == 'mlp_bn':
            self.gate = torch.nn.Sequential(torch.nn.Linear(2048*num_ensem, 1024),
                                                    torch.nn.BatchNorm1d(1024),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    torch.nn.Linear(1024, 512),
                                                    torch.nn.BatchNorm1d(512),
                                                    # torch.nn.ReLU(inplace=True),  # second layer
                                                    torch.nn.Linear(512, num_ensem)
                                                    )  # output layer
        elif gate_arch == 'mlp_bn3':
            self.gate = torch.nn.Sequential(torch.nn.Linear(2048*num_ensem, 1024),
                                                    torch.nn.BatchNorm1d(1024),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    torch.nn.Linear(1024, 1024),
                                                    torch.nn.BatchNorm1d(1024),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    torch.nn.Linear(1024, 512),
                                                    torch.nn.BatchNorm1d(512),
                                                    # torch.nn.ReLU(inplace=True),  # second layer
                                                    torch.nn.Linear(512, num_ensem)
                                                    )  # output layer
        elif gate_arch == 'mlp_bn4':
            self.gate = torch.nn.Sequential(torch.nn.Linear(2048*num_ensem, 1024),
                                                    torch.nn.BatchNorm1d(1024),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    torch.nn.Linear(1024, 1024),
                                                    torch.nn.BatchNorm1d(1024),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    torch.nn.Linear(1024, 1024),
                                                    torch.nn.BatchNorm1d(1024),
                                                    torch.nn.ReLU(inplace=True),
                                                    torch.nn.Linear(1024, 512),
                                                    torch.nn.BatchNorm1d(512),
                                                    # torch.nn.ReLU(inplace=True),  # second layer
                                                    torch.nn.Linear(512, num_ensem)
                                                    )  # output layer
        elif gate_arch == 'mlp_bn4w':
            self.gate = torch.nn.Sequential(torch.nn.Linear(2048*num_ensem, 2048),
                                                    torch.nn.BatchNorm1d(2048),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    torch.nn.Linear(2048, 2048),
                                                    torch.nn.BatchNorm1d(2048),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    torch.nn.Linear(2048, 1024),
                                                    torch.nn.BatchNorm1d(1024),
                                                    torch.nn.ReLU(inplace=True),
                                                    torch.nn.Linear(1024, 512),
                                                    torch.nn.BatchNorm1d(512),
                                                    # torch.nn.ReLU(inplace=True),  # second layer
                                                    torch.nn.Linear(512, num_ensem)
                                                    )  # output layer

        elif gate_arch == 'smallmlp':
            self.gate = torch.nn.Sequential(torch.nn.Linear(2048*num_ensem, smallmlp_hd),
                                                    # torch.nn.LayerNorm(1024),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    # torch.nn.Linear(512, 128),
                                                    # torch.nn.LayerNorm(512),
                                                    # torch.nn.ReLU(inplace=True),  # second layer
                                                    torch.nn.Linear(smallmlp_hd, num_ensem)
                                                    )  # output layer
        elif gate_arch == 'smallmlp_bn':
            self.gate = torch.nn.Sequential(torch.nn.Linear(2048*num_ensem, smallmlp_hd),
                                                    torch.nn.BatchNorm1d(smallmlp_hd),
                                                    torch.nn.ReLU(inplace=True),  # first layer
                                                    # torch.nn.Linear(512, 128),
                                                    # torch.nn.LayerNorm(512),
                                                    # torch.nn.ReLU(inplace=True),  # second layer
                                                    torch.nn.Linear(smallmlp_hd, num_ensem),
                                                    torch.nn.BatchNorm1d(smallmlp_hd)
                                                    )  # output layer
        elif 'resnet18' in gate_arch:
            self.gate = models_dict['resnet18'](num_classes=num_ensem)
        elif 'resnet50' in gate_arch:
            self.gate = models_dict['resnet50'](num_classes=num_ensem)
        elif gate_arch == 'smallcnn':
            self.gate = resnet0(num_classes=num_ensem)
        elif gate_arch == 'vit_tiny':
            self.gate = vit_tiny(patch_size=vit_patch_size)
        elif gate_arch == 'vit_small':
            self.gate = vit_small(patch_size=vit_patch_size)
        elif gate_arch == 'vit_base':
            self.gate = vit_base(patch_size=vit_patch_size)
        else:
            raise "gate_arch not defined"

        if 'att' in gate_arch:
            self.gate.fc = nn.Identity()

        self.set_eval_mode(eval_mode)
        self.gate_arch = gate_arch


    def load_sep_weights(self, weights_path_list, gate_path=None):
        for m in range(self.num_ensem):
            weights = weights_path_list[m]
            state_dict = torch.load(weights, map_location='cpu')

            cur_mem = self.members[m]
            if 'model' in state_dict:
                consume_prefix_in_state_dict_if_present(state_dict['model'], 'members.0.') # this is assuming only 1 member was trained at a time
                cur_mem.load_state_dict(state_dict['model'], strict=True)
                print("===> Loaded backbone state dict from ", weights)
            elif 'backbone' in state_dict:
                missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["backbone"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)

            if 'log_reg_weight' in state_dict and 'log_reg_bias' in state_dict:
                cur_mem.fc.weight = torch.from_numpy(state_dict['log_reg_weight'])
                cur_mem.fc.bias = torch.from_numpy(state_dict['log_reg_bias'])
        if gate_path is not None:
            gate_state_dict = torch.load(gate_path, map_location='cpu')['state_dict']
            new_gate_state_dict = {k.replace("layers.", ""):v for k,v in gate_state_dict.items()}
            self.gate.load_state_dict(new_gate_state_dict)
            print("===> Loaded gate state dict from ", gate_path)

    def load_weights(self, weights_path_list, convert_from_single=False):
        # convert_from_single: whether to convert the weights from 1 model (MultiBackbone)
        # making sure that the number of pretrained weights & ensem member size is equal
        if not convert_from_single:
            assert len(self.members) == len(weights_path_list)
        else:
            assert len(weights_path_list) == 1
            ensem_state_dict = torch.load(weights_path_list[0], map_location='cpu')
            # ensem_state_dict = ensem_state_dict['enc_state_dict']
            state_dicts = convert_weights_from_single_backbone(ensem_state_dict, self.num_ensem)

        for m in range(self.num_ensem):
            cur_mem = self.members[m]

            if convert_from_single:
                weights = ''
                state_dict = state_dicts[m]
            else:
                weights = weights_path_list[m]
                print(weights.split("/")[-1])
                state_dict = torch.load(weights, map_location='cpu')

            if 'epoch' in str(weights):
                consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.backbone.')
                cur_mem.load_state_dict(state_dict['model'], strict=False)
            elif 'simsiam' in str(weights):
                consume_prefix_in_state_dict_if_present(state_dict['state_dict'], 'module.backbone.')
                consume_prefix_in_state_dict_if_present(state_dict['state_dict'], 'backbone.')
                missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["state_dict"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)
            else:
                missing_keys, unexpected_keys = cur_mem.load_state_dict(state_dict["backbone"], strict=False)
                print('missing keys', missing_keys)
                print('unexpected keys', unexpected_keys)
                # assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []

            if self.eval_mode in {'linear_probe', 'finetune'}:
                cur_mem.fc.weight.data.normal_(mean=0.0, std=0.01)
                cur_mem.fc.bias.data.zero_()

    def set_eval_mode(self, mode='freeze'):
        self.eval_mode = mode
        if self.eval_mode == 'all_freeze':
            for cur_mem in self.members:
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(False)
            self.gate.requires_grad_(False)
        if self.eval_mode == 'freeze':
            for cur_mem in self.members:
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(False)
            self.gate.requires_grad_(True)
        elif self.eval_mode == 'linear_probe':
            for cur_mem in self.members:
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(True)
            self.gate.requires_grad_(True)
        elif self.eval_mode == 'finetune':
            for cur_mem in self.members:
                cur_mem.requires_grad_(True)
                cur_mem.fc.requires_grad_(True)
        elif self.eval_mode == 'log_reg':
            for cur_mem in self.members:
                cur_mem.fc = nn.Identity()
                cur_mem.requires_grad_(False)
                cur_mem.fc.requires_grad_(False)
        else:
            raise NotImplementedError(f'Evaluation mode {mode} not implemented')

    # forward that feeds inputs to the ensemble and returns stacked outputs
    def forward(self, x, gate_cond='x', top_1=False, loss_fn='ce'):
        outputs = []
        if gate_cond == 'none':
            for cur_mem in self.members:
                output = cur_mem(x)
                outputs.append(output)
            return torch.stack(outputs)

        elif gate_cond == 'x':
            if 'att' not in self.gate_arch:
                weighting = self.gate(x)
                if loss_fn == 'ce' or 'softmax' in loss_fn:
                    weighting = weighting.softmax(dim=-1)
                elif 'sig' in loss_fn:
                    weighting = torch.nn.functional.sigmoid(weighting)
                elif 'tanh' in loss_fn:
                    weighting = (torch.tanh(weighting) + 1)/2

                for i,cur_mem in enumerate(self.members):
                    output = cur_mem(x,return_rep=False)
                    outputs.append(output)
            else:
                reps = []
                gate_output = self.gate(x).unsqueeze(1)
                if 'cos' in self.gate_arch:
                    gate_output = torch.nn.functional.normalize(gate_output, dim=-1)
                for i,cur_mem in enumerate(self.members):
                    rep, output = cur_mem(x,return_both=True)
                    if 'cos' in self.gate_arch:
                        rep = torch.nn.functional.normalize(rep, dim=-1)
                    reps.append(rep)
                    outputs.append(output)
                reps = torch.stack(reps,dim=2)
                weighting = torch.matmul(gate_output,reps)
                if 'scaled' in self.gate_arch:
                    weighting /= (gate_output.size(2) ** 0.5)
                weighting = weighting.squeeze(1)
                if loss_fn == 'ce' or 'softmax' in loss_fn:
                    weighting = weighting.softmax(dim=-1)
                elif 'sig' in loss_fn:
                    weighting = torch.nn.functional.sigmoid(weighting)
                elif 'tanh' in loss_fn:
                    weighting = (torch.tanh(weighting) + 1)/2

        elif gate_cond == 'z':
            reps = []
            for i,cur_mem in enumerate(self.members):
                rep = cur_mem(x,return_rep=True)
                reps.append(rep)
                output = cur_mem.fc(rep)
                outputs.append(output)
            if 'vit' in self.gate_arch:
                weighting = self.gate(reps)
                if loss_fn == 'ce' or 'softmax' in loss_fn:
                    weighting = weighting.softmax(dim=-1)
                elif 'sig' in loss_fn:
                    weighting = torch.nn.functional.sigmoid(weighting)
                elif 'tanh' in loss_fn:
                    weighting = (torch.tanh(weighting) + 1)/2
            else:
                weighting = self.gate(torch.cat(reps, dim=1))
                if loss_fn == 'ce' or 'softmax' in loss_fn:
                    weighting = weighting.softmax(dim=-1)
                elif 'sig' in loss_fn:
                    weighting = torch.nn.functional.sigmoid(weighting)
                elif 'tanh' in loss_fn:
                    weighting = (torch.tanh(weighting) + 1)/2

        if top_1:
            top1_logits, top1_indices = weighting.topk(1,dim=1)
            top1_gates = top1_logits.softmax(dim=1)
            zeros = torch.zeros_like(weighting, requires_grad=True).to(weighting.device)
            weighting = zeros.scatter(1, top1_indices, top1_gates)

        return torch.stack(outputs), weighting

    # forward that feeds inputs to the ensemble and returns only the averaged outputs from the ensemble
    def forward_ensem(self, x):
        outputs = self.forward(x)

        return outputs.softmax(dim=-1).mean(dim=0)

models_dict = {
    'resnet18': models.resnet18,
    'cifar_resnet18': cifar_resnet.resnet18,
    'resnet50': models.resnet50,
    'mlp': MLP,
}
