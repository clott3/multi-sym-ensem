#!/usr/bin/env python

import os
import random

classes_r = os.listdir('/data/scratch/swhan/data/imagenet-r/')
classes_a = os.listdir('/data/scratch/swhan/data/imagenet-a/')
classes_a = [c for c in classes_a if os.path.isdir('/data/scratch/swhan/data/imagenet-a/'+c)]
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

classes_com = intersection(classes_r, classes_a)
print(len(classes_com))

classes_r_left = [c for c in classes_r if c not in classes_com]
add_classes = random.sample(classes_r_left, 100-len(classes_com))
classes_com = classes_com + add_classes
classes_com = set(classes_com)
print(len(classes_com))

classes_chosen = random.sample(classes_com, 100)

with open('./in100_classes.txt', 'w+') as f:
    for c in classes_chosen:
        f.write(f'{c}\n')
