from itertools import chain
import os
import torch
import torch.nn as nn
from torchvision import transforms

from data import paired_dataloader, custom_dataloader
from models import Generator, Discriminator
from utils import init_weights, LossTracker, try_gpu, image_transform, denormalize_image, save_image, ImageBuffer

# Parameters
LEARNING_RATE = 2e-4
SAVE_STEPS = 25
SAVE_IMAGE_STEPS = 10
BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, 1, 30, 30)
LOSS_NAMES = ['G_A_identity_loss', 'G_B_identity_loss', 'G_A_loss', 'G_B_loss', 'G_A_cycle_loss', 'G_B_cycle_loss']
EPOCHS = 200

MSE_Loss = nn.MSELoss()
L1_Loss = nn.L1Loss()
D_real_target = torch.ones(SHAPE, device=try_gpu(), requires_grad=False)
D_fake_target = torch.zeros(SHAPE, device=try_gpu(), requires_grad=False)

# image transforms
image_transforms = [transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

MODEL_PATH = './model_ckpts'
GAN_IMG_PATH = './gan_imgs'
EPOCH_IMG_PATH = './epoch_imgs' # list of images to run and save after each epoch
PATH_A = './monet_jpg'
PATH_B = './land_imgs/thumbnail/train'
dataloader = paired_dataloader(PATH_A, PATH_B, BATCH_SIZE, transforms_=image_transforms)
epoch_dataloader = custom_dataloader(EPOCH_IMG_PATH, BATCH_SIZE, transforms_=image_transforms[1:])

# Create computation graphs for Models
# G_A is Generator that maps from A domain to B domain
# D_A is Discriminates whether or not an image is from A domain
G_A = Generator()
G_B = Generator()
D_A = Discriminator()
D_B = Discriminator()

# Move Models to GPU
G_A.cuda()
G_B.cuda()
D_A.cuda()
D_B.cuda()

# Initialize Weights to N(0,0.2)
G_A.apply(init_weights)
G_B.apply(init_weights)
D_A.apply(init_weights)
D_B.apply(init_weights)

# Create Adam Optimizers
adam_cycle = torch.optim.Adam(chain(G_A.parameters(), G_B.parameters()), lr=LEARNING_RATE, betas=(0.5, 0.999))
adam_D_A = torch.optim.Adam(D_A.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
adam_D_B = torch.optim.Adam(D_B.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

def linear_decay(num_epochs, decay_epoch):
    """
    Returns a function that returns a weight for linear decay after decay_epoch
    """
    assert num_epochs > decay_epoch
    def decay_func(epoch):
        if epoch < decay_epoch:
            return 1.0
        else:
            return 1.0 - (epoch - decay_epoch) / (num_epochs - decay_epoch)
    return decay_func

lr_decay = linear_decay(200, 100)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(adam_cycle, lr_lambda=lr_decay)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(adam_D_A, lr_lambda=lr_decay)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(adam_D_B, lr_lambda=lr_decay)

# Mixed Precision Training and Gradient Scaling
scaler = torch.cuda.amp.GradScaler()


def cycle_update(G_A, G_B, D_A, D_B, A_img, B_img, optimizer):
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        # Identity Loss
        B_identity = G_A(B_img)
        B_identity_loss = 5.0 * L1_Loss(B_identity, B_img)

        A_identity = G_B(A_img)
        A_identity_loss = 5.0 * L1_Loss(A_identity, A_img)

        # Generator Loss
        # Generator is trying to trick Discriminator labeling generated image as 1
        fake_B = G_A(A_img)
        fake_A = G_B(B_img)
        
        # Don't update Discriminator when updating Generator
        # with torch.no_grad():
        fake_B_output = D_B(fake_B)
        fake_A_output = D_A(fake_A)
        
        fake_B_loss = MSE_Loss(fake_B_output, D_real_target)
        fake_A_loss = MSE_Loss(fake_A_output, D_real_target)

        # Cycle Loss
        cycle_A_output = G_B(fake_B)
        cycle_A_loss = 10.0 * MSE_Loss(cycle_A_output, A_img)

        cycle_B_output = G_A(fake_A)
        cycle_B_loss = 10.0 * MSE_Loss(cycle_B_output, B_img)

        total_loss = A_identity_loss + B_identity_loss + fake_B_loss + fake_A_loss + cycle_A_loss + cycle_B_loss
    
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    # total_loss.backward()
    # optimizer.step()
    return (B_identity_loss + cycle_B_loss + fake_B_loss).item(), (A_identity_loss + cycle_A_loss + fake_A_loss).item()

def discriminator_update(D, real_imgs, fake_imgs, optimizer):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        fake_output = D(fake_imgs)
        real_output = D(real_imgs)
        discriminator_loss = (MSE_Loss(real_output, D_real_target) + MSE_Loss(fake_output, D_fake_target)) * 0.5
    
    scaler.scale(discriminator_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    # discriminator_loss.backward()
    # optimizer.step()
    return discriminator_loss.item()

tracker = LossTracker(['G_A Loss', 'G_B Loss', 'D_A Loss', 'D_B Loss'], BATCH_SIZE)
A_buffer = ImageBuffer(50)
B_buffer = ImageBuffer(50)

for e in range(EPOCHS):
    count = 0
    for (A_filename, A_img, B_filename, B_img) in dataloader:
        B_filename = B_filename[0]
        A_img = A_img.to(device=try_gpu())
        B_img = B_img.to(device=try_gpu())
        # Cycle Update
        G_A_loss, G_B_loss = cycle_update(G_A, G_B, D_A, D_B, A_img, B_img, adam_cycle)
        
        # Create Fake Image to update Discriminator
        fake_A = A_buffer.query(G_B(B_img).detach())
        D_A_loss = discriminator_update(D_A, A_img, fake_A, adam_D_A)
        fake_B = B_buffer.query(G_A(A_img).detach())
        D_B_loss = discriminator_update(D_B, B_img, fake_B, adam_D_B)

        tracker.add([G_A_loss, G_B_loss, D_A_loss, D_B_loss])

    # Log Losses
    print('Epoch {} | G_A Loss: {:.2f} | G_B Loss: {:.2f} | D_A Loss: {:.2f} | D_B Loss: {:.2f}'.format(
        e, *tracker.get_loss()))
    tracker.reset()

    if (e+1) % SAVE_IMAGE_STEPS == 0:
        for (filename, img) in epoch_dataloader:
            name = 'epoch{}_'.format(e) + filename[0]
            with torch.no_grad():
                generated_img = G_B(img.to(device=try_gpu()))
                save_image(denormalize_image(generated_img), os.path.join(GAN_IMG_PATH, name))
    
    # update learning rate
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    if (e+1) % SAVE_STEPS == 0:
        model_name = 'ganlands_model_epoch{}.pt'.format(e)
        model_dict = {
                      'epoch': e+1,
                      'G_A':G_A.state_dict(),
                      'G_B':G_B.state_dict(),
                      'D_A':D_A.state_dict(),
                      'D_B':D_B.state_dict()
                    }
        torch.save(model_dict, os.path.join(MODEL_PATH, model_name))
        




