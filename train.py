from itertools import chain
import torch
import torch.nn as nn

from data import paired_dataloader
from models import Generator, Discriminator
from utils import init_weights, LossTracker, try_gpu, image_transform

# Parameters
LEARNING_RATE = 2e-5
SAVE_STEPS = 25
BATCH_SIZE = 8
SHAPE = (BATCH_SIZE, 1, 30, 30)
PATH_A = './monet_jpg'
PATH_B = './land_imgs/thumbnail'
dataloader = paired_dataloader(PATH_A, PATH_B, BATCH_SIZE, transform=image_transform)
LOSS_NAMES = ['G_A_identity_loss', 'G_B_identity_loss', 'G_A_loss', 'G_B_loss', 'G_A_cycle_loss', 'G_B_cycle_loss']
EPOCHS = 100

MSE_Loss = nn.MSELoss()
L1_Loss = nn.L1Loss()
D_real_target = torch.ones(SHAPE, device=try_gpu(), requires_grad=False)
D_fake_target = torch.zeros(SHAPE, device=try_gpu(), requires_grad=False)

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

# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

def cycle_update(G_A, G_B, D_A, D_B, A_img, B_img, optimizer):
    optimizer.zero_grad()

    # Identity Loss
    B_identity = G_A(B_img)
    B_identity_loss = L1_Loss(B_identity, B_img)

    A_identity = G_B(A_img)
    A_identity_loss = L1_Loss(A_identity, A_img)

    # Generator Loss
    # Generator is trying to trick Discriminator labeling generated image as 1
    fake_B = G_A(A_img)
    fake_A = G_B(B_img)
    
    # Don't update Discriminator when updating Generator
    with torch.no_grad():
        fake_B_output = D_B(fake_B)
        fake_A_output = D_A(fake_A)
    
    fake_B_loss = MSE_Loss(fake_B_output, D_real_target)
    fake_A_loss = MSE_Loss(fake_A_output, D_real_target)

    # Cycle Loss
    cycle_A_output = G_B(fake_B)
    cycle_A_loss = MSE_Loss(cycle_A_output, A_img)

    cycle_B_output = G_A(fake_A)
    cycle_B_loss = MSE_Loss(cycle_B_output, B_img)

    total_loss = A_identity_loss + B_identity_loss + fake_B_loss + fake_A_loss + cycle_A_loss + cycle_B_loss
    total_loss.backward()
    optimizer.step()
    return [A_identity_loss.item(), B_identity_loss.item(), fake_A_loss.item(), fake_B_loss.item(), cycle_A_loss.item(), cycle_B_loss.item()]

def discriminator_update(D, real_imgs, fake_imgs, optimizer):
    optimizer.zero_grad()
    fake_output = D(fake_imgs)
    real_output = D(real_imgs)
    discriminator_loss = (MSE_Loss(real_output, D_real_target) + MSE_Loss(fake_output, D_fake_target)) * 0.5
    discriminator_loss.backward()
    optimizer.step()
    return discriminator_loss.item()

tracker = LossTracker(['G_A Loss', 'G_B Loss', 'D_A Loss', 'D_B Loss'], BATCH_SIZE)

for e in range(EPOCHS):
    for A_img, B_img in dataloader:
        A_img = A_img.to(device=try_gpu())
        B_img = B_img.to(device=try_gpu())
        # Cycle Update
        cycle_update_losses = cycle_update(G_A, G_B, D_A, D_B, A_img, B_img, adam_cycle)
        
        # Create Fake Image to update Discriminator
        fake_A = G_B(B_img).detach()
        D_A_loss = discriminator_update(D_A, A_img, fake_A, adam_D_A)
        fake_B = G_A(A_img).detach()
        D_B_loss = discriminator_update(D_B, B_img, fake_B, adam_D_B)

        # print([cycle_update_losses[2].item(), cycle_update_losses[3].item(), D_A_loss.item(), D_B_loss.item()])
        tracker.add([cycle_update_losses[2], cycle_update_losses[3], D_A_loss, D_B_loss])

    # Log Losses
    print('Epoch {} | G_A Loss: {:.2f} | G_B Loss: {:.2f} | D_A Loss: {:.2f} | D_B Loss: {:.2f}'.format(
        e, *tracker.get_loss())
    )

    tracker.reset()
    
    if (e+1) % SAVE_STEPS == 0:
        model_dict = {
                      'epoch': e+1,
                      'G_A':G_A.state_dict(),
                      'G_B':G_B.state_dict(),
                      'D_A':D_A.state_dict(),
                      'D_B':D_B.state_dict()
                    }
        torch.save(model_dict, PATH)




