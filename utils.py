import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
import random
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

class ImageBuffer():
    """
    Update Discriminator network using image buffer rather than
    current Generator output to reduce model ocillation 
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.images = []

    def query(self, imgs):
        return_images = []
        for img in imgs:
            img = torch.unsqueeze(img, 0)
            if len(self.images) < self.buffer_size:
                self.images.append(img)
                return_images.append(img)
            else:
                p = random.uniform(0, 1)
                if p > 0.5: # return random image from buffer and replace with new image
                    idx = random.randint(0, self.buffer_size - 1)
                    rand_buffer_img = self.images[idx].clone()
                    self.images[idx] = img
                    return_images.append(rand_buffer_img)
                else: # return input image
                    return_images.append(img)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
                    
