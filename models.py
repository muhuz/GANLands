import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    ResNet Block Implementation for the Generator Network
    """
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        layers = [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels, in_channels, 3),
                  nn.InstanceNorm2d(in_channels),
                  nn.ReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels, in_channels, 3),
                  nn.InstanceNorm2d(in_channels)]
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, X):
        return X + self.block(X)

class Generator(nn.Module):
    def __init__(self):
        """
        The Generator used in CycleGAN takes in a tensor of shape (3, 256, 256)
        and returns a generated image of the same size.
        """
        super(Generator, self).__init__()
        
        # Downsampling Layers
        
        # Shape: (64, 256, 256)
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(3, 64, kernel_size=7),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True)]
        
        # Shape: (128, 128, 128)
        layers += [
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)]

        # Shape: (256, 64, 64)
        layers += [
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)]

        # ResNet Block Layers

        # Shape: (256, 64, 64)
        layers += [ResBlock(256) for _ in range(6)]

        # Upsampling Layers

        # Shape: (128, 128, 128)
        layers += [nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(128),
                   nn.ReLU(inplace=True)]

        # Shape: (64, 256, 256)
        layers += [nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(64),
                   nn.ReLU(inplace=True)]
        
        # Output Layers

        # Final Shape: (3, 256, 256)
        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(64, 3, 7),
                   nn.Tanh()]

        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Shape: (64, 128, 128)
        layers = [nn.Conv2d(3, 64, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        
        # Shape: (128, 64, 64)
        layers += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(128),
                   nn.LeakyReLU(0.2, inplace=True)]

        # Shape: (256, 32, 32)
        layers += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(256),
                   nn.LeakyReLU(0.2, inplace=True)]
        
        # Shape: (512, 31, 31)
        layers += [nn.Conv2d(256, 512, 4, padding=1),
                   nn.InstanceNorm2d(512),
                   nn.LeakyReLU(0.2, inplace=True)]
        
        # Final Shape: (1, 30, 30)
        layers += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X)


