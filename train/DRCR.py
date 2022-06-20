import torch
from torch import nn
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class Conv2D(nn.Module):
    def __init__(self, in_channel=256, out_channel=8):
        super(Conv2D, self).__init__()
        self.guide_conv2D = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        spatial_guidance = self.guide_conv2D(x)
        return spatial_guidance

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            pass
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class NPM(nn.Module):
    def __init__(self, in_channel):
        super(NPM, self).__init__()
        self.in_channel = in_channel
        self.activation = nn.LeakyReLU(0.2, inplace = True)
        self.conv0_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv0_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_0_cat = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.conv2_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv2_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_2_cat = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.conv4_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv4_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_4_cat = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)

        self.conv_cat = nn.Conv2d(in_channel*3, in_channel, 3, 1, 1)
    
    def forward(self, x):

        x_0 = x
        x_2 = F.avg_pool2d(x, 2, 2)
        x_4 = F.avg_pool2d(x_2, 2, 2)

        x_0 = torch.cat([self.conv0_33(x_0), self.conv0_11(x_0)], 1)
        x_0 = self.activation(self.conv_0_cat(x_0))

        x_2 = torch.cat([self.conv2_33(x_2), self.conv2_11(x_2)], 1)
        x_2 = F.interpolate(self.activation(self.conv_2_cat(x_2)), scale_factor=2, mode='bilinear')

        x_4 = torch.cat([self.conv2_33(x_4), self.conv2_11(x_4)], 1)
        x_4 = F.interpolate(self.activation(self.conv_4_cat(x_4)), scale_factor=4, mode='bilinear')

        x = x + self.activation(self.conv_cat(torch.cat([x_0, x_2, x_4], 1)))
        return x

class CRM(nn.Module):
    def __init__(self, channel, reduction = 8):
        super(CRM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DRCR_Block(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(DRCR_Block, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv3 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv4 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv5 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv6 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        # self.cspn2_guide = GMLayer(in_channels)
        # self.cspn2 = Affinity_Propagate_Channel()
        self.se1 = CRM(in_channels)
        self.se2 = CRM(in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # guidance2 = self.cspn2_guide(x3)
        # x3_2 = self.cspn2(guidance2, x3)
        x3_2 = self.se1(x)
        x4 = self.conv4(torch.cat((x3, x3_2), 1))
        x5 = self.conv5(torch.cat((x2, x4), 1))
        x6 = self.conv6(torch.cat((x1, x5), 1))+self.se2(x3_2)
        return x6

class DRCR(nn.Module):
    def __init__(self, inplanes=3, planes=31, channels=200, n_DRBs=8):
        super(DRCR, self).__init__()
        self.input_conv2D = Conv3x3(inplanes, channels, 3, 1)
        self.input_prelu2D = nn.PReLU()
        self.head_conv2D = Conv3x3(channels, channels, 3, 1)
        self.denosing = NPM(channels)
        self.backbone = nn.ModuleList(
            [DRCR_Block(channels, channels) for _ in range(n_DRBs)])
        self.tail_conv2D = Conv3x3(channels, channels, 3, 1)
        self.output_prelu2D = nn.PReLU()
        self.output_conv2D = Conv3x3(channels, planes, 3, 1)

    def forward(self, x):
        out = self.DRN2D(x)
        return out

    def DRN2D(self, x):
        out = self.input_prelu2D(self.input_conv2D(x))
        out = self.head_conv2D(out)
        out = self.denosing(out)

        for i, block in enumerate(self.backbone):
            out = block(out)

        out = self.tail_conv2D(out)
        out = self.output_conv2D(self.output_prelu2D(out))
        return out






if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_tensor = torch.rand(1, 3, 128, 128)
    model = DRCR(3, 31, 100, 10)
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    print(output_tensor.size())
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print(torch.__version__)





