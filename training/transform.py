from torchvision.transforms import functional as F
import torchvision.transforms as T

import random
import numpy as np
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            print("Applied Horizontal Flip")
            image = F.hflip(image)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = image.width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

class ColorJitter(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            print("Applied Color Jitter")
            image = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(image)
        return image, target

class AddGaussianNoise(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            print("Applied Gaussian Noise")
            np_image = np.array(image)
            mean = 0
            sigma = 25
            gauss = np.random.normal(mean, sigma, np_image.shape)
            noisy = np.clip(np_image + gauss, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy)
        return image, target

def get_transform(train):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip())
        transforms.append(ColorJitter())
        transforms.append(AddGaussianNoise())
    return Compose(transforms)