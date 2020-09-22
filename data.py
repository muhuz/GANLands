from itertools import cycle
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transforms_=None):
        self.root_dir = root_dir
        if isinstance(transforms_, list):
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = transforms_
        self.filenames = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.filenames[idx]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return filename, image

class PairedDataset(Dataset):
    def __init__(self, A_root_dir, B_root_dir, transforms_=None):
        self.A_root_dir = A_root_dir
        self.B_root_dir = B_root_dir
        if isinstance(transforms_, list):
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = transforms_
        self.A_filenames = os.listdir(A_root_dir)
        self.B_filenames = os.listdir(B_root_dir)
        self.A_size = len(self.A_filenames)
        self.B_size = len(self.B_filenames)
    
    def __len__(self):
        return max(self.A_size, self.B_size)
    
    def __getitem__(self, idx):
        A_filename = self.A_filenames[idx % self.A_size]
        B_filename = self.B_filenames[random.randint(0, self.B_size - 1)]
        A_img = Image.open(os.path.join(self.A_root_dir, A_filename))
        B_img = Image.open(os.path.join(self.B_root_dir, B_filename))
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        return A_filename, A_img, B_filename, B_img

def paired_dataloader(A_root_dir, B_root_dir, batch_size, transforms_=None, shuffle=True, num_workers=0):
    dataset = PairedDataset(A_root_dir, B_root_dir, transforms_=transforms_)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def custom_dataloader(root_dir, batch_size, transforms_=None, shuffle=True, num_workers=0):
    dataset = CustomDataset(root_dir, transforms_=transforms_)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# def paired_dataloader(path_A, path_B, batch_size, transforms_=None, shuffle=True, num_workers=0):
#     """
#     Create paired dataloader that cycles the smaller dataset
#     """
#     dataset_A = CustomDataset(path_A, transforms_=transforms_)
#     dataset_B = CustomDataset(path_B, transforms_=transforms_)
#     if len(dataset_A) > len(dataset_B):
#         dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=shuffle)
#         dataloader_B = cycle(DataLoader(dataset_B, batch_size=batch_size, shuffle=shuffle))
#     else:
#         dataloader_A = cycle(DataLoader(dataset_A, batch_size=batch_size, shuffle=shuffle))
#         dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=shuffle)
#     return zip(dataloader_A, dataloader_B)

if __name__ == "__main__":
    monet_data = './monet_jpg'
    land_data = './land_imgs/thumbnail'
    names = os.listdir(monet_data)
    img = Image.open(os.path.join(monet_data, names[0]))
    dataset = PairedDataset(monet_data, land_data)
    dataloader = paired_dataloader(monet_data, land_data, 1, transforms_=transforms.ToTensor())
    for e in range(2):
        counter = 0
        for i, (A_filename, A_img, B_filename, B_img) in enumerate(dataloader):
            print(A_img)
            counter += 1
            break