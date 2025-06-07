import torch
import torch.nn as nn
from utils.conv_blocks import ConvBlock, DeconvBlock

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),                  
            ConvBlock(64, 128, stride=2),      
            ConvBlock(128, 256, stride=2),     
            ConvBlock(256, 512, stride=2),     
            ConvBlock(512, 512, stride=2),   
        )
        
        self.decoder = nn.Sequential(
            DeconvBlock(512, 512),           
            DeconvBlock(512, 256),            
            DeconvBlock(256, 128),             
            DeconvBlock(128, 64),              
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon
