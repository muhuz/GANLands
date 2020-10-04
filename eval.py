import os
import torch
import torch.nn as nn
from torchvision import transforms

from data import custom_dataloader
from models import Generator
from utils import denormalize_image, save_image

TRAIN_IMG_PATH = './land_imgs/thumbnail/train'
TEST_IMG_PATH = './land_imgs/thumbnail/test'

TRAIN_SAVE_IMG_PATH = './eval_imgs/new_results/train'
TEST_SAVE_IMG_PATH = './eval_imgs/new_results/test'

MODEL_PATH = './model_ckpts/ganlands_model_epoch199.pt'
BATCH_SIZE = 1
image_transforms = [transforms.ToTensor(), 
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
train_dataloader = custom_dataloader(TRAIN_IMG_PATH, BATCH_SIZE, transforms_=image_transforms)
test_dataloader = custom_dataloader(TEST_IMG_PATH, BATCH_SIZE, transforms_=image_transforms)

# restore model
G_A = Generator()
G_B = Generator()
model_dict = torch.load(MODEL_PATH)
G_A.load_state_dict(model_dict['G_A'])
G_B.load_state_dict(model_dict['G_B'])
G_A.eval()
G_B.eval()

for (filename, img) in train_dataloader:
    generated_img = G_B(img).detach()
    cycle_img = G_A(generated_img).detach()
    save_image(denormalize_image(generated_img), os.path.join(TRAIN_SAVE_IMG_PATH, filename[0]))
    save_image(denormalize_image(cycle_img), os.path.join(TRAIN_SAVE_IMG_PATH, 'cycle_' + filename[0]))
    print('saved')

for (filename, img) in test_dataloader:
    generated_img = G_B(img).detach()
    cycle_img = G_A(generated_img).detach()
    save_image(denormalize_image(generated_img), os.path.join(TEST_SAVE_IMG_PATH, filename[0]))
    save_image(denormalize_image(cycle_img), os.path.join(TEST_SAVE_IMG_PATH, 'cycle_' + filename[0]))


