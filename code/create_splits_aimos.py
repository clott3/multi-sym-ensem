import torch
import numpy as np
import torchvision
import os
import json

dataset='imagenet'
data_path = '/gpfs/u/locker/200/CADS/datasets/ImageNet'
train_dataset = torchvision.datasets.ImageFolder(f'{data_path}/train')
val_dataset = torchvision.datasets.ImageFolder(f'{data_path}/val')

num_per_class = {}
# count total number of samples per class:
for i in train_dataset.targets:
    if i not in num_per_class:
        num_per_class[i] = 0
    else:
        num_per_class[i] += 1
print(num_per_class)
#
# num_classes = [100,200,500,800]
#
# for ncls in num_classes:
#     os.makedirs(f'./calib_splits/', exist_ok=True)
#     if os.path.exists(f'./calib_splits/am_IN{ncls}_with_INlabels_train.txt'):
#         os.remove(f'./calib_splits/am_IN{ncls}_with_INlabels_train.txt')
#     if os.path.exists(f'./calib_splits/am_IN{ncls}_with_INlabels_val.txt'):
#         os.remove(f'./calib_splits/am_IN{ncls}_with_INlabels_val.txt')
#
#     for i, (filepath, target) in enumerate(train_dataset.samples):
#         if target < ncls:
#             line = '/'.join(filepath.split('/')[-3:])
#             with open(f'./calib_splits/am_IN{ncls}_with_INlabels_train.txt', "a") as f: # unlabeled includes labeled set
#                 f.write(line+f" {target}\n")
#     print("created train split for num_classes = ", ncls)
#
#     for i, (filepath, target) in enumerate(val_dataset.samples):
#         if target < ncls:
#             line = '/'.join(filepath.split('/')[-3:])
#             with open(f'./calib_splits/am_IN{ncls}_with_INlabels_val.txt', "a") as f: # unlabeled includes labeled set
#                 f.write(line+f" {target}\n")
#     print("created val split for num_classes = ", ncls)
#
#     # check number of samples
#     with open(f'./calib_splits/am_IN{ncls}_with_INlabels_train.txt', "r") as f2:
#         for count, line in enumerate(f2):
#             pass
#         total_num1 = count + 1
#     print(f"total in train split: ", total_num1)
#     with open(f'./calib_splits/am_IN{ncls}_with_INlabels_val.txt', "r") as f2:
#         for count, line in enumerate(f2):
#             pass
#         total_num2 = count + 1
#     print(f"total in val split: ", total_num2)
# class_to_idx = train_dataset.class_to_idx
# print(len(class_to_idx), len(train_dataset.samples))


num_classes = [100,200,500,800]

for ncls in num_classes:
    chosen_classes = np.random.choice(np.arange(1000), ncls, replace=False)
    label_map = {in_lab:new_lab for in_lab,new_lab in zip(chosen_classes,np.arange(ncls))}
    str_label_map = {str(k):str(v) for k,v in label_map.items()}
    with open(f'./calib_splits/am_IN{ncls}_INlabel_to_ds_map.json', "w") as outfile:
        json.dump(str_label_map, outfile)

    os.makedirs(f'./calib_splits/', exist_ok=True)
    if os.path.exists(f'./calib_splits/am_IN{ncls}_train.txt'):
        os.remove(f'./calib_splits/am_IN{ncls}_train.txt')
    if os.path.exists(f'./calib_splits/am_IN{ncls}_val.txt'):
        os.remove(f'./calib_splits/am_IN{ncls}_val.txt')

    for i, (filepath, target) in enumerate(train_dataset.samples):
        if target in chosen_classes:
            line = '/'.join(filepath.split('/')[-3:])
            with open(f'./calib_splits/am_IN{ncls}_train.txt', "a") as f: # unlabeled includes labeled set
                f.write(line+f" {label_map[target]}\n")
    print("created train split for num_classes = ", ncls)

    for i, (filepath, target) in enumerate(val_dataset.samples):
        if target in chosen_classes:
            line = '/'.join(filepath.split('/')[-3:])
            with open(f'./calib_splits/am_IN{ncls}_val.txt', "a") as f: # unlabeled includes labeled set
                f.write(line+f" {label_map[target]}\n")
    print("created val split for num_classes = ", ncls)

    # check number of samples
    with open(f'./calib_splits/am_IN{ncls}_train.txt', "r") as f2:
        for count, line in enumerate(f2):
            pass
        total_num1 = count + 1
    print(f"total in train split: ", total_num1)
    with open(f'./calib_splits/am_IN{ncls}_val.txt', "r") as f2:
        for count, line in enumerate(f2):
            pass
        total_num2 = count + 1
    print(f"total in val split: ", total_num2)
class_to_idx = train_dataset.class_to_idx
print(len(class_to_idx), len(train_dataset.samples))

#
# for val_percent in [5,10,20]:
#     val_num_per_class = {clsind: int(val_percent/100*num_sam) for clsind, num_sam in num_per_class.items()}
#     print(val_num_per_class)
#     lab_dict = {}
#     os.makedirs(f'./calib_splits/', exist_ok=True)
#     if os.path.exists(f'./calib_splits/am_{dataset}_{val_percent}percent_train.txt'):
#         os.remove(f'./calib_splits/am_{dataset}_{val_percent}percent_train.txt')
#     if os.path.exists(f'./calib_splits/am_{dataset}_{val_percent}percent_val.txt'):
#         os.remove(f'./calib_splits/am_{dataset}_{val_percent}percent_val.txt')
#
#     val_count = {clsind: 0 for clsind in range(len(class_to_idx))}
#     for i, (filepath, target) in enumerate(train_dataset.samples):
#         line = '/'.join(filepath.split('/')[-3:])
#         if val_count[target] <= val_num_per_class[target]:
#             with open(f'./calib_splits/am_{dataset}_{val_percent}percent_val.txt', "a") as f: # unlabeled includes labeled set
#                 f.write(line+f" {target}\n")
#             val_count[target] += 1
#         else:
#             with open(f'./calib_splits/am_{dataset}_{val_percent}percent_train.txt', "a") as f: # unlabeled includes labeled set
#                 f.write(line+f" {target}\n")
#     print("created split for percent = ", val_percent)
#
#     # check number of samples
#     with open(f'./calib_splits/am_{dataset}_{val_percent}percent_val.txt', "r") as f2:
#         for count, line in enumerate(f2):
#             pass
#         total_num1 = count + 1
#     print(total_num1)
#     with open(f'./calib_splits/am_{dataset}_{val_percent}percent_train.txt', "r") as f2:
#         for count, line in enumerate(f2):
#             pass
#         total_num2 = count + 1
#     print(total_num2)
#     assert(total_num1+total_num2 == len(train_dataset))


# fix 50k val set
# val_num_per_class = {clsind: 50 for clsind, num_sam in num_per_class.items()}
# print(val_num_per_class)
# lab_dict = {}
# os.makedirs(f'./calib_splits/', exist_ok=True)
# if os.path.exists(f'./calib_splits/am_{dataset}_val50k_train.txt'):
#     os.remove(f'./calib_splits/am_{dataset}_val50k_train.txt')
# if os.path.exists(f'./calib_splits/am_{dataset}_val50k_val.txt'):
#     os.remove(f'./calib_splits/am_{dataset}_val50k_val.txt')
#
# val_count = {clsind: 0 for clsind in range(len(class_to_idx))}
# for i, (filepath, target) in enumerate(train_dataset.samples):
#     line = '/'.join(filepath.split('/')[-3:])
#     if val_count[target] <= val_num_per_class[target]:
#         with open(f'./calib_splits/am_{dataset}_val50k_val.txt', "a") as f: # unlabeled includes labeled set
#             f.write(line+f" {target}\n")
#         val_count[target] += 1
#     else:
#         with open(f'./calib_splits/am_{dataset}_val50k_train.txt', "a") as f: # unlabeled includes labeled set
#             f.write(line+f" {target}\n")
# print("created split for 50k")
#
# # check number of samples
# with open(f'./calib_splits/am_{dataset}_val50k_val.txt', "r") as f2:
#     for count, line in enumerate(f2):
#         pass
#     total_num1 = count + 1
# print(total_num1)
# with open(f'./calib_splits/am_{dataset}_val50k_train.txt', "r") as f2:
#     for count, line in enumerate(f2):
#         pass
#     total_num2 = count + 1
# print(total_num2)
# assert(total_num1+total_num2 == len(train_dataset))

# val_percent = 20
# val_num_per_class = {clsind: int(val_percent/100*num_sam) for clsind, num_sam in num_per_class.items()}
# print(val_num_per_class)
# for fold in [0,1,2,3,4]:
#     lab_dict = {}
#     os.makedirs(f'./calib_splits/', exist_ok=True)
#     if os.path.exists(f'./calib_splits/am_{dataset}_5uniqfolds_train_fold{fold}.txt'):
#         os.remove(f'./calib_splits/am_{dataset}_5uniqfolds_train_fold{fold}.txt')
#     if os.path.exists(f'./calib_splits/am_{dataset}_5uniqfolds_val_fold{fold}.txt'):
#         os.remove(f'./calib_splits/am_{dataset}_5uniqfolds_val_fold{fold}.txt')
#
# val_count_fold0 = {clsind: 0 for clsind in range(len(class_to_idx))}
# val_count_fold1 = {clsind: 0 for clsind in range(len(class_to_idx))}
# val_count_fold2 = {clsind: 0 for clsind in range(len(class_to_idx))}
# val_count_fold3 = {clsind: 0 for clsind in range(len(class_to_idx))}
# val_count_fold4 = {clsind: 0 for clsind in range(len(class_to_idx))}
#
# for i, (filepath, target) in enumerate(train_dataset.samples):
#     line = '/'.join(filepath.split('/')[-3:])
#     if val_count_fold0[target] <= val_num_per_class[target]:
#         with open(f'./calib_splits/am_{dataset}_5uniqfolds_val_fold0.txt', "a") as f: # unlabeled includes labeled set
#             f.write(line+f" {target}\n")
#         val_count_fold0[target] += 1
#         for fold in [1,2,3,4]:
#             with open(f'./calib_splits/am_{dataset}_5uniqfolds_train_fold{fold}.txt', "a") as f: # unlabeled includes labeled set
#                 f.write(line+f" {target}\n")
#     elif val_count_fold1[target] <= val_num_per_class[target]:
#         with open(f'./calib_splits/am_{dataset}_5uniqfolds_val_fold1.txt', "a") as f: # unlabeled includes labeled set
#             f.write(line+f" {target}\n")
#         val_count_fold1[target] += 1
#         for fold in [0,2,3,4]:
#             with open(f'./calib_splits/am_{dataset}_5uniqfolds_train_fold{fold}.txt', "a") as f: # unlabeled includes labeled set
#                 f.write(line+f" {target}\n")
#     elif val_count_fold2[target] <= val_num_per_class[target]:
#         with open(f'./calib_splits/am_{dataset}_5uniqfolds_val_fold2.txt', "a") as f: # unlabeled includes labeled set
#             f.write(line+f" {target}\n")
#         val_count_fold2[target] += 1
#         for fold in [0,1,3,4]:
#             with open(f'./calib_splits/am_{dataset}_5uniqfolds_train_fold{fold}.txt', "a") as f: # unlabeled includes labeled set
#                 f.write(line+f" {target}\n")
#     elif val_count_fold3[target] <= val_num_per_class[target]:
#         with open(f'./calib_splits/am_{dataset}_5uniqfolds_val_fold3.txt', "a") as f: # unlabeled includes labeled set
#             f.write(line+f" {target}\n")
#         val_count_fold3[target] += 1
#         for fold in [0,1,2,4]:
#             with open(f'./calib_splits/am_{dataset}_5uniqfolds_train_fold{fold}.txt', "a") as f: # unlabeled includes labeled set
#                 f.write(line+f" {target}\n")
#     else:
#         with open(f'./calib_splits/am_{dataset}_5uniqfolds_val_fold4.txt', "a") as f: # unlabeled includes labeled set
#             f.write(line+f" {target}\n")
#         val_count_fold4[target] += 1
#         for fold in [0,1,2,3]:
#             with open(f'./calib_splits/am_{dataset}_5uniqfolds_train_fold{fold}.txt', "a") as f: # unlabeled includes labeled set
#                 f.write(line+f" {target}\n")
# print("created split for 5 folds! Now checking.. ")
#
# # check number of samples
# for fold in [0,1,2,3,4]:
#     with open(f'./calib_splits/am_{dataset}_5uniqfolds_val_fold{fold}.txt', "r") as f2:
#         for count, line in enumerate(f2):
#             pass
#         total_num1 = count + 1
#     with open(f'./calib_splits/am_{dataset}_5uniqfolds_train_fold{fold}.txt', "r") as f2:
#         for count, line in enumerate(f2):
#             pass
#         total_num2 = count + 1
#     print(total_num2)
#     print(f"fold {fold}: total val = {total_num1}, total train= {total_num2}, grand total={total_num1+total_num2}")
#     assert(total_num1+total_num2 == len(train_dataset))
