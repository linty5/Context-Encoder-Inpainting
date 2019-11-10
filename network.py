import torch
import torch.nn as nn
from network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
#
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # The generator is U shaped
        # It means: input -> downsample -> upsample -> output
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 4, 2, 1, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E5 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Bottleneck
        self.B1 = Conv2dLayer(opt.start_channels * 8, opt.bottleneck_channels, 4, pad_type = opt.pad, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.bottleneck_channels, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = Conv2dLayer(opt.start_channels, opt.out_channels, 4, 2, 1, pad_type = opt.pad, norm = 'none', activation = 'tanh')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        x = self.E1(x)                                          # out: batch * 64 * 64 * 64
        x = self.E2(x)                                          # out: batch * 64 * 32 * 32
        x = self.E3(x)                                          # out: batch * 128 * 16 * 16
        x = self.E3(x)                                          # out: batch * 256 * 8 * 8
        x = self.E3(x)                                          # out: batch * 512 * 4 * 4
        # Bottleneck
        x = self.B1(x)                                          # out: batch * 4000 * 1 * 1
        # Decode the center code
        x = self.D1(x)                                          # out: batch * 128 * 128 * 128
        x = self.D2(x)                                          # out: batch * 64 * 256 * 256
        x = self.D3(x)                                          # out: batch * out_channel * 256 * 256
        x = self.D4(x)                                          # out: batch * 128 * 128 * 128
        x = self.D5(x)                                          # out: batch * 64 * 256 * 256

        return x

# ----------------------------------------
#       AdversarialDiscriminator
# ----------------------------------------
# Usage:

class AdversarialDiscriminator(nn.Module):

    def __init__(self, opt):
        super(AdversarialDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.out_channels, 64, 4, 2, 1, pad_type=opt.pad, norm='none', sn=True)
        self.block2 = Conv2dLayer(64, 128, 4, 2, 1, pad_type=opt.pad, norm=opt.norm, sn=True)
        self.block3 = Conv2dLayer(128, 256, 4, 2, 1, pad_type=opt.pad, norm=opt.norm, sn=True)
        self.block4 = Conv2dLayer(256, 512, 4, 2, 1, pad_type=opt.pad, norm=opt.norm, sn=True)
        self.final1 = Conv2dLayer(512, 512, 4, 2, 1, pad_type=opt.pad, norm=opt.norm, sn=True)
        self.final2 = Conv2dLayer(512, 1, 3, 1, 1, pad_type=opt.pad, norm='none', activation='none', sn=True)

    def forward(self, x):
        x = self.block1(x)  # out: batch * 64 * 64 * 64
        x = self.block2(x)  # out: batch * 64 * 32 * 32
        x = self.block3(x)  # out: batch * 128 * 16 * 16
        x = self.block4(x)  # out: batch * 256 * 8 * 8
        x = self.final1(x)  # out: batch * 512 * 4 * 4
        x = self.final2(x)  # out: batch * 1 * 4 * 4
        return x
