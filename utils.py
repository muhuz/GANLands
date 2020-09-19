import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader

MONET_PATH = './monet_jpg'

def image_transform(image):
    return torch.tensor(image.reshape(3, 256, 256), dtype=torch.float)

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
        self.loss_history = [[] * len(labels)]
        self.curr_loss = [0] * len(labels)
        self.curr_n = 0
        self.batch_size = batch_size
    
    def add(self, losses):
        assert len(losses) == len(self.labels)
        for i in range(len(losses)):
            self.curr_loss[i] += losses[i]
        self.curr_n += self.batch_size
    
    def reset(self):
        for i in range(len(curr_loss)):
            self.loss_history[i].append(self.curr_loss[i] / self.curr_n)
            self.curr_loss[i] = 0
        self.curr_n = 0
    
    def get_loss(self):
        return self.curr_loss

    def get_history(self):
        return dict(zip(self.labels, self.loss_history))


# if __name__ == "__main__":
#     dataset = MonetDataset('monet_jpg')
#     fig = plt.figure()
#     for i in range(5):
#         sample = dataset[i]
#         ax = plt.subplot(1, 5, i+1)
#         plt.tight_layout()
#         ax.set_title('Sample #{}'.format(i))
#         ax.axis('off')
#         plt.imshow(sample)
#     plt.show()

