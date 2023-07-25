from re import M
import torch
import torch.nn as nn
from torchvision import models

import collections

from utils import consume_prefix_in_state_dict_if_present
import sys

class EnsembleSSL(nn.Module):
    def __init__(self, arch, num_ensem=1, num_classes=1000, eval_mode='freeze'):
        super().__init__()
        print("initialized EnsembleSSL")
        self.num_ensem = num_ensem
        self.num_classes = num_classes
        model_fn = models_dict[arch]
        self.members = torch.nn.ModuleList([model_fn(num_classes=self.num_classes) for _ in range(num_ensem)])
        self.set_eval_mode(eval_mode)

    def load_weights(self, weights_path_list):
        print("loading weights")
        for m in range(self.num_ensem):
            weights = weights_path_list[m]
            state_dict = torch.load(weights, map_location='cpu')

            cur_mem = self.members[m]
            if 'model' in state_dict:
                if 'members.0.fc.weight' in state_dict['model']:
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
        else:
            raise NotImplementedError(f'Evaluation mode {mode} not implemented')


    # forward that feeds inputs to the ensemble and returns stacked outputs
    def forward(self, x):
        outputs = []

        for cur_mem in self.members:
            output = cur_mem(x)
            outputs.append(output)

        return torch.stack(outputs)

    # forward that feeds inputs to the ensemble and returns only the averaged outputs from the ensemble
    def forward_ensem(self, x):
        outputs = self.forward(x)

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


models_dict = {
    'resnet50': models.resnet50,
    'mlp': MLP,
}
