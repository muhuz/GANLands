import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader

MONET_PATH = './monet_jpg'

class MonetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image

def MonetDataLoader(batch_size, shuffle=True, num_workers=0):
    return DataLoader(MonetDataset(MONET_PATH, transform=reshape_monet), batch_size=batch_size, num_workers=num_workers)

def reshape_monet(image):
    return image.reshape(3, 256, 256).astype(float)

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
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

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

