import torch
import torchvision
import numpy as np
import os.path as osp
from PIL import Image
import torchvision
import torchvision.transforms as TF

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def vae_transforms(split, aug='randcrop', img_size=256):
    t = []
    if split == 'train':
        if aug == 'randcrop':
            t.append(TF.Resize(img_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True))
            t.append(TF.RandomCrop(img_size))
        elif aug == 'centercrop':
            t.append(TF.Lambda(lambda x: center_crop_arr(x, img_size)))
        else:
            raise ValueError(f"Invalid augmentation: {aug}")
        t.append(TF.RandomHorizontalFlip(p=0.5))
    else:
        t.append(TF.Lambda(lambda x: center_crop_arr(x, img_size)))
        
    t.append(TF.ToTensor())

    return TF.Compose(t)


def cached_transforms(aug='tencrop', img_size=256, crop_ranges=[1.05, 1.10]):
    t = []
    if 'centercrop' in aug:
        t.append(TF.Lambda(lambda x: center_crop_arr(x, img_size)))
        t.append(TF.Lambda(lambda x: torch.stack([TF.ToTensor()(x), TF.ToTensor()(TF.functional.hflip(x))])))
    elif 'tencrop' in aug:
        crop_sizes = [int(img_size * crop_range) for crop_range in crop_ranges]
        t.append(TF.Lambda(lambda x: [center_crop_arr(x, crop_size) for crop_size in crop_sizes]))
        t.append(TF.Lambda(lambda crops: [crop for crop_tuple in [TF.TenCrop(img_size)(crop) for crop in crops] for crop in crop_tuple]))
        t.append(TF.Lambda(lambda crops: torch.stack([TF.ToTensor()(crop) for crop in crops])))
    else:
        raise ValueError(f"Invalid augmentation: {aug}")

    return TF.Compose(t)

class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root, split='train', aug='randcrop', img_size=256):
        super().__init__(osp.join(root, split))
        if not 'cache' in aug:
            self.transform = vae_transforms(split, aug=aug, img_size=img_size)
        else:
            self.transform = cached_transforms(aug=aug, img_size=img_size)