import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from itertools import permutations

def jigsaw_images(imgs, gpu, number=1, trans_p=0.5):
    a = [0, 1, 2, 3]
    perms = list(permutations(a))
    p_vec = np.ones(24) * trans_p/23
    p_vec[0] = 1 - trans_p
    p_vec = list(p_vec)
    # imgs: BxCxHxW
    b, c, h, w = imgs.shape
    if number == 1:
        # labels = []
        for i in range(b):
            a1 = imgs[i][:, :h // 2, :w // 2]
            a2 = imgs[i][:, :h // 2, w // 2:]
            a3 = imgs[i][:, h // 2:, w // 2:]
            a4 = imgs[i][:, h // 2:, :w // 2]

            jigperm = np.random.choice(24, 1, p=p_vec).item()
            # labels.append(jigperm)
            permuted_stack = torch.stack([a1, a2, a3, a4])[list(perms[jigperm])]
            imgs[i] = torch.cat([torch.cat([permuted_stack[0], permuted_stack[1]], dim=-1),
                                 torch.cat([permuted_stack[3], permuted_stack[2]], dim=-1)], dim=-2)
        # return imgs.cuda(gpu, non_blocking=True), torch.LongTensor(labels).cuda(gpu, non_blocking=True)
        return imgs

    else:
        n_jigsaw_images = number * b
        jig_images = torch.zeros([n_jigsaw_images, imgs.shape[1], imgs.shape[2], imgs.shape[3]]).cuda(gpu, non_blocking=True)
        jig_classes = torch.zeros([n_jigsaw_images]).long().cuda(gpu, non_blocking=True)
        pchoice = np.random.choice(24, number, replace=False) # make sure different ones sampled
        a1 = imgs[:, :, :h // 2, :w // 2]
        a2 = imgs[:,:, :h // 2, w // 2:]
        a3 = imgs[:,:, h // 2:, w // 2:]
        a4 = imgs[:, :, h // 2:, :w // 2]

        for i, p in enumerate(pchoice):
            permuted_stack = torch.stack([a1, a2, a3, a4])[list(perms[p])]
            jig_images[i*b: (i+1)*b] = torch.cat([torch.cat([permuted_stack[0], permuted_stack[1]], dim=-1),
                                 torch.cat([permuted_stack[3], permuted_stack[2]], dim=-1)], dim=-2)
            jig_classes[i*b: (i+1)*b] = p

        return jig_images, jig_classes

# rotation
def rotate_images(images, gpu, number=4, trans_p=0.5):
    nimages = images.shape[0]

    if number == 1:
        # y = []
        for i in range(nimages):
            rotdeg = np.random.choice(4, 1, p=[(1-trans_p), trans_p/3, trans_p/3, trans_p/3]).item()
            # y.append(rotdeg)
            images[i] = torch.rot90(images[i], rotdeg, [1, 2])
        # y = torch.LongTensor(y).cuda(gpu, non_blocking=True)
        # return images.cuda(gpu, non_blocking=True), y
        return images

    elif number == -1:
        y = []
        for i in range(nimages):
            rotdeg = np.random.choice(4, 1).item()
            y.append(rotdeg)
            images[i] = torch.rot90(images[i], rotdeg, [1, 2])
        y = torch.LongTensor(y).cuda(gpu, non_blocking=True)
        # return images.cuda(gpu, non_blocking=True), y
        return images, y


    n_rot_images = 4 * nimages
    # rotate images all 4 ways at once
    rotated_images = torch.zeros([n_rot_images, images.shape[1], images.shape[2], images.shape[3]]).cuda(gpu,
                                                                                                         non_blocking=True)
    rot_classes = torch.zeros([n_rot_images]).long().cuda(gpu, non_blocking=True)

    rotated_images[:nimages] = images
    # rotate 90
    rotated_images[nimages:2 * nimages] = images.flip(3).transpose(2, 3)
    rot_classes[nimages:2 * nimages] = 1
    # rotate 180
    rotated_images[2 * nimages:3 * nimages] = images.flip(3).flip(2)
    rot_classes[2 * nimages:3 * nimages] = 2
    # rotate 270
    rotated_images[3 * nimages:4 * nimages] = images.transpose(2, 3).flip(3)
    rot_classes[3 * nimages:4 * nimages] = 3

    return rotated_images, rot_classes


def blur_images(images, gpu, number=4, trans_p = 0.5):
    nimages = images.shape[0]
    k = [0, 5, 9, 15]
    p_vec = np.ones(4) * trans_p/3
    p_vec[0] = 1 - trans_p
    p_vec = list(p_vec)

    if number == 1:
        for i in range(nimages):
            j = np.random.choice(4, 1, p=p_vec).item()
            if j == 0:
                continue
            else:
                images[i] = TF.gaussian_blur(images[i], k[j])
        return images

    nimages = images.shape[0]
    n_rot_images = 4 * nimages

    # rotate images all 4 ways at once
    rotated_images = torch.zeros([n_rot_images, images.shape[1], images.shape[2], images.shape[3]]).cuda(gpu, non_blocking=True)
    rot_classes = torch.zeros([n_rot_images]).long().cuda(gpu, non_blocking=True)

    rotated_images[:nimages] = images
    rotated_images[nimages:2 * nimages] = TF.gaussian_blur(images, 5)
    rot_classes[nimages:2 * nimages] = 1
    rotated_images[2 * nimages:3 * nimages] = TF.gaussian_blur(images, 9)
    rot_classes[2 * nimages:3 * nimages] = 2
    rotated_images[3 * nimages:4 * nimages] = TF.gaussian_blur(images, 15)
    rot_classes[3 * nimages:4 * nimages] = 3

    return rotated_images, rot_classes


def solarize_images(images, gpu, number=4, trans_p=0.5):
    nimages = images.shape[0]
    k = [0, 85, 170, 256]
    p_vec = np.ones(4) * trans_p/3
    p_vec[0] = 1 - trans_p
    p_vec = list(p_vec)

    if number == 1:
        for i in range(nimages):
            j = np.random.choice(4, 1, p=p_vec).item()
            if j == 0: continue
                # images[i] = images[i]
            else:
                images[i] = TF.solarize(images[i], k[j])
            # return TF.solarize(images, k[j])
        return images

    n_rot_images = 4 * nimages

    # rotate images all 4 ways at once
    rotated_images = torch.zeros([n_rot_images, images.shape[1], images.shape[2], images.shape[3]]).cuda(gpu, non_blocking=True)
    rot_classes = torch.zeros([n_rot_images]).long().cuda(gpu, non_blocking=True)

    rotated_images[:nimages] = TF.solarize(images, 0)
    rotated_images[nimages:2 * nimages] = TF.solarize(images, 85)
    rot_classes[nimages:2 * nimages] = 1
    rotated_images[2 * nimages:3 * nimages] = TF.solarize(images, 170)
    rot_classes[2 * nimages:3 * nimages] = 2
    rotated_images[3 * nimages:4 * nimages] = TF.solarize(images, 256)
    rot_classes[3 * nimages:4 * nimages] = 3

    return rotated_images, rot_classes

class RandomInvert(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        # _log_api_usage_once(self)
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be inverted.

        Returns:
            PIL Image or Tensor: Randomly color inverted image.
        """
        if torch.rand(1).item() < self.p:
            return TF.invert(img)
        return img

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])
vertical_flip_transform = T.RandomVerticalFlip(1.0)
horizontal_flip_transform = T.RandomHorizontalFlip(1.0)
invert_transform = RandomInvert(1.0)
grayscale_transform = T.RandomGrayscale(1.0)

def halfswap(imgs):
    # imgs: BxCxHxW
    if len(imgs.shape) == 3:
        c, h, w = imgs.shape
        imgs = torch.cat([imgs[:, h // 2:, :], imgs[:, :h // 2, :]], dim=-2)
    elif len(imgs.shape) == 4:
        b, c, h, w = imgs.shape
        imgs = torch.cat([imgs[:, :, h // 2:, :], imgs[:, :, :h // 2, :]], dim=-2)
    return imgs

def halfswap_images(images, gpu, number=1, trans_p=0.5):
    nimages = images.shape[0]

    if number == 1:
        pvec = [1-trans_p, trans_p]
        for i in range(nimages):
            j = np.random.choice(2,1,p=pvec).item()
            if j == 0:
                continue
            else:
                images[i] = halfswap(images[i])

        return images

    n_flip_images = 2 * nimages

    flip_images = torch.zeros([n_flip_images, images.shape[1], images.shape[2], images.shape[3]]).cuda(gpu, non_blocking=True)
    flip_classes = torch.zeros([n_flip_images]).long().cuda(gpu, non_blocking=True)

    flip_images[:nimages] = images

    flip_images[nimages:2 * nimages] = halfswap(images)
    flip_classes[nimages:2 * nimages] = 1

    return flip_images, flip_classes


def grayscale_images(images, gpu, number=1, trans_p=0.5):
    nimages = images.shape[0]
    if number == 1:
        pvec = [1-trans_p, trans_p]
        for i in range(nimages):

            j = np.random.choice(2,1,p=pvec).item()
            if j == 0:
                continue
            else:
                images[i] = grayscale_transform(images[i])
        return images

    n_flip_images = 2 * nimages

    flip_images = torch.zeros([n_flip_images, images.shape[1], images.shape[2], images.shape[3]]).cuda(gpu, non_blocking=True)
    flip_classes = torch.zeros([n_flip_images]).long().cuda(gpu, non_blocking=True)

    flip_images[:nimages] = images

    flip_images[nimages:2 * nimages] = grayscale_transform(images)
    flip_classes[nimages:2 * nimages] = 1

    return flip_images, flip_classes

def invert_images(images, gpu, number=1, trans_p=0.5):
    nimages = images.shape[0]

    if number == 1:
        pvec = [1-trans_p, trans_p]
        for i in range(nimages):

            j = np.random.choice(2,1,p=pvec).item()
            if j == 0:
                continue
            else:
                images[i] = invert_transform(images[i])

        return images

    n_flip_images = 2 * nimages

    flip_images = torch.zeros([n_flip_images, images.shape[1], images.shape[2], images.shape[3]]).cuda(gpu, non_blocking=True)
    flip_classes = torch.zeros([n_flip_images]).long().cuda(gpu, non_blocking=True)

    flip_images[:nimages] = images

    flip_images[nimages:2 * nimages] = invert_transform(images)
    flip_classes[nimages:2 * nimages] = 1

    return flip_images, flip_classes

def vertical_flip_images(images, gpu, number=1, trans_p=0.5):
    nimages = images.shape[0]

    if number == 1:
        pvec = [1-trans_p, trans_p]
        for i in range(nimages):
            j = np.random.choice(2,1,p=pvec).item()
            if j == 0:
                continue
            else:
                images[i] = vertical_flip_transform(images[i])
        return images

    n_flip_images = 2 * nimages

    flip_images = torch.zeros([n_flip_images, images.shape[1], images.shape[2], images.shape[3]]).cuda(gpu, non_blocking=True)
    flip_classes = torch.zeros([n_flip_images]).long().cuda(gpu, non_blocking=True)

    flip_images[:nimages] = images

    flip_images[nimages:2 * nimages] = vertical_flip_transform(images)
    flip_classes[nimages:2 * nimages] = 1

    return flip_images, flip_classes


choose_transform = {'jigsaw': jigsaw_images,
                    'rotate': rotate_images,
                    'blur': blur_images,
                    'solarize': solarize_images,
                    'vflip': vertical_flip_images,
                    'invert': invert_images,
                    'halfswap': halfswap_images,
                    'grayscale': grayscale_images}
num_tfm_classes = {'jigsaw': 24,
                    'rotate': 4,
                    'blur': 4,
                    'solarize': 4,
                    'vflip': 2,
                    'invert': 2,
                    'halfswap': 2,
                    'grayscale': 2}
