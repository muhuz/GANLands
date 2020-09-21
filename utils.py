import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MONET_PATH = './monet_jpg'

def image_transform(image):
    transforms = [transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    return transforms

def denormalize_image(tensor):
    """
    Transform tensor output with entries in [-1,1] to an image with entries in [0, 255]
    """
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    return image.astype(np.uint8)

def save_image(array, path):
    img = Image.fromarray(array.transpose(1, 2, 0), 'RGB')
    img.save(path)

def view_image(path):
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class LossTracker():
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.loss_history = [[] for i in range(len(labels))]
        self.curr_loss = [0] * len(labels)
        self.curr_n = 0
        self.batch_size = batch_size
    
    def add(self, losses):
        assert len(losses) == len(self.labels)
        for i in range(len(losses)):
            self.curr_loss[i] += losses[i]
        self.curr_n += self.batch_size
    
    def reset(self):
        for i in range(len(self.curr_loss)):
            self.loss_history[i].append(self.curr_loss[i] / self.curr_n)
            self.curr_loss[i] = 0
        self.curr_n = 0
    
    def get_loss(self):
        return [loss / self.curr_n for loss in self.curr_loss]

    def get_history(self):
        return dict(zip(self.labels, self.loss_history))
