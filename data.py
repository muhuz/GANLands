from itertools import cycle
import os
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
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

def paired_dataloader(path_A, path_B, batch_size, transform=None, shuffle=True, num_workers=0):
    """
    Create paired dataloader that cycles the smaller dataset
    """
    dataset_A = CustomDataset(path_A, transform=transform)
    dataset_B = CustomDataset(path_B, transform=transform)
    if len(dataset_A) > len(dataset_B):
        dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=shuffle)
        dataloader_B = cycle(DataLoader(dataset_B, batch_size=batch_size, shuffle=shuffle))
    else:
        dataloader_A = cycle(DataLoader(dataset_A, batch_size=batch_size, shuffle=shuffle))
        dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=shuffle)
    return zip(dataloader_A, dataloader_B)

if __name__ == "__main__":
    monet_data = './monet_jpg'
    land_data = './land_imgs/thumbnail'
    dataloader = paired_dataloader(monet_data, land_data, 1)
    for i, (imgA, imgB) in enumerate(dataloader):
        print(i)