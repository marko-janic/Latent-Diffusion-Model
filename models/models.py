from codecs import latin_1_encode
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, image_channels=1):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.'RRRRRRRRRRR
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(image_channels, channels[0], 3, stride=1, bias=False, padding=1)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False, padding=1)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False, padding=1)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False, padding=1)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, padding=1, output_padding=1)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, padding=1, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, padding=1, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, padding=1, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding='same')
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding='same')
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Unet(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024), image_size=None):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)

        self.chs         = chs
        # self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.upconvs    = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear')for i in range(len(chs)-1)])
        self.dec_blocks1 = nn.ModuleList([Block(chs[-1-i], chs[-1-(i+1)]) for i in range(len(chs)-1)]) 
        self.dec_blocks2 = nn.ModuleList([Block(chs[-1-(i+1)]*2, chs[-1-(i+1)]) for i in range(len(chs)-1)]) 

        self.image_size = image_size
        if image_size!=None:
            self.latent_dim = int(image_size/(2**(len(chs)-1)))
            self.mlp = nn.Linear(self.latent_dim**2*chs[-1] , self.latent_dim**2*chs[-1])
    
    def forward(self, x):
        # encoder
        ftrs = []
        ftrs.append(x)
        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)
            ftrs.append(x)

        # mlp en latent space
        if self.image_size != None:
            x = torch.flatten(x, 1)
            x = self.mlp(x)
            # x = F.relu(x)
            x = x.reshape([-1, self.chs[-1], self.latent_dim, self.latent_dim])

        # decoder
        # import ipdb; ipdb.set_trace()
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            x        = self.dec_blocks1[i](x)
            x        = torch.cat([x, ftrs[len(self.chs)-1-i-1]], dim=1)
            x        = self.dec_blocks2[i](x)
        return x


class ScoreUnet(nn.Module):
    def __init__(self, marginal_prob_std, chs=(3,64,128,256,512,1024), image_size=None, embed_dim=256):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
        self.embed_blocks1 = nn.ModuleList([nn.Linear(embed_dim, chs[i]) for i in range(len(chs))])
        self.embed_blocks2 = nn.ModuleList([nn.Linear(embed_dim, chs[i]) for i in range(len(chs))])

        self.chs         = chs
        # self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.upconvs    = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear')for i in range(len(chs)-1)])
        self.dec_blocks1 = nn.ModuleList([Block(chs[-1-i], chs[-1-(i+1)]) for i in range(len(chs)-1)]) 
        self.dec_blocks2 = nn.ModuleList([Block(chs[-1-(i+1)]*2, chs[-1-(i+1)]) for i in range(len(chs)-1)]) 

        self.image_size = image_size
        if image_size!=None:
            self.latent_dim = int(image_size/(2**(len(chs)-1)))
            self.mlp = nn.Linear(self.latent_dim**2*chs[-1] , self.latent_dim**2*chs[-1])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))

        # For normalization of the output
        self.marginal_prob_std = marginal_prob_std
    
    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))    

        # encoder
        ftrs = []
        ftrs.append(x)
        i = 1
        for block in self.enc_blocks:
            x = block(x)
            y = self.embed_blocks1[i](embed)[..., None, None]
            x = self.pool(x+y)
            ftrs.append(x)
            i+=1

        # mlp en latent space
        # import ipdb; ipdb.set_trace()
        if self.image_size != None:
            x = torch.flatten(x, 1)
            x = self.mlp(x)
            # x = F.relu(x)
            x = x.reshape([-1, self.chs[-1], self.latent_dim, self.latent_dim])

        # decoder
        # import ipdb; ipdb.set_trace()
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            x        = self.dec_blocks1[i](x)
            y        = self.embed_blocks2[-2-i](embed)[..., None, None]
            x        = torch.cat([x+y, ftrs[len(self.chs)-1-i-1]], dim=1)
            x        = self.dec_blocks2[i](x)

        # Normalize output
        x = x / self.marginal_prob_std(t)[:, None, None, None]

        return x

