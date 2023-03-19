import torch
import numpy as np

for ver in ['train', 'val']:
    txt = f'./stylized-imagenet-{ver}.txt'
    img_paths = []
    labels = []
    img_dict = {}
    with open(txt) as f:
        for line in f:
            img1 = line.split()[0]
            label = int(line.split()[1])
            sty_label = int(line.split()[2])
            img_dict[sty_label] = img1
            if sty_label == 4:
                img_paths.append(img_dict)
                labels.append(label)
                img_dict = {}
    img_paths = np.array(img_paths)
    labels = np.array(labels)
    np.save(f"./sty_IN_{ver}_img_paths.npy", img_paths)
    np.save(f"./sty_IN_{ver}_labels.npy", labels)
