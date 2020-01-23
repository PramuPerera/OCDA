'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=72):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out), out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()




import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes=72, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), out



import os
import sys
import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import autograd
from torchvision import models
from torch.autograd import Variable

import pdb

class classifier32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*4*4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = x.view(batch_size, -1)

        return self.fc1(x), x

class Dense(nn.Module):
    def __init__(self, nt, gpu_ids=[]):
        super(Dense, self).__init__()
        self.ef_dim = nt
        self.relu = nn.ReLU(inplace=True)
        ############# 64x64  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.norm0 = haze_class.features.norm0
        self.relu0 = haze_class.features.relu0
        self.pool0 = haze_class.features.pool0

        ############# Block1-down 16x16  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 8x8  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  4x4 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        ############# Block4-up  512x2x2  ##############
        self.hr_joint = nn.Sequential(
            conv3x3(nt+512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.residual = self._make_layer(ResBlock, 512)

        self.bottle_block4 = BottleneckBlock(512, 256)
        self.trans_block4 = TransitionBlock(768, 128)

        ############# Block5-up  4x4 ##############
        self.bottle_block5 = BottleneckBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)

        ############# Block6-up 8x8   ##############
        self.bottle_block6 = BottleneckBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)

        ############# Block7-up 16x16   ##############
        self.bottle_block7 = BottleneckBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)

        ## 128 X  128
        ############# Block8-up c 32x32  ##############
        self.bottle_block8 = BottleneckBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.tanh = nn.Tanh()

        def _make_layer(self, block, channel_num):
            layers = []
            for i in range(2):
                layers.append(block(channel_num))
            return nn.Sequential(*layers)

        def forward(self, x, attr):
            ## 256x256
            x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))
            ## 64 X 64
            x1 = self.dense_block1(x0)
            x1 = self.trans_block1(x1)
            ###  32x32
            x2 = self.trans_block2(self.dense_block2(x1))
            ### 16 X 16
            x3 = self.trans_block3(self.dense_block3(x2))
            attr = attr.view(-1, self.ef_dim,1,1)
            attr = attr.repeat(1,1,2,2)
            xf = self.hr_joint(torch.cat([x3, attr],1))
            xf = self.residual(xf)
            ## 8 X 8
            x4 = self.trans_block4(self.bottle_block4(xf))
            x42 = torch.cat([x4, x2], 1)
            ## 16 X 16
            x5 = self.trans_block5(self.bottle_block5(x42))
            x52 = torch.cat([x5, x1], 1)
            ##  32 X 32
            x6 = self.trans_block6(self.bottle_block6(x52))
            ##  64 X 64
            x7 = self.trans_block7(self.bottle_block7(x6))
            ##  128 X 128
            x8 = self.trans_block8(self.bottle_block8(x7))
            x8 = torch.cat([x8, x], 1)
            x9 = self.relu(self.conv_refin(x8))
            out = self.tanh(self.refine3(x9))

            return out

def define_E(attrB, sketch_nc, nz, nt, ngf, gpu_ids=[]):
    netE1 = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    netE1 = E1(sketch_nc, nz, nt, ngf, attrB, gpu_ids=gpu_ids)
    if len(gpu_ids) > 0:
        netE1.cuda()
    netE1.apply(weights_init)
    return netE1

def define_G( attrB, attrA, sketch_nc, image_nc, nz, nt, ngf, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    netG1 = G1(sketch_nc, nz, nt, ngf, attrB, gpu_ids=gpu_ids)
    netG2 = Dense(nt, gpu_ids=gpu_ids)
    netG3 = ConUnetGenerator(image_nc, image_nc, attrB+attrA, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG1.cuda()
        netG2.cuda()
        netG3.cuda()
    netG1.apply(weights_init)
    netG2.apply(weights_init)
    netG3.apply(weights_init)
    return netG1, netG2, netG3

def define_D(attrB, attrA, sketch_nc, image_nc, nz, nt, ndf, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    print(gpu_ids)
    netD2 = NLayerDiscriminator(sketch_nc+image_nc, ndf, n_layers = 4, norm_layer = nn.BatchNorm2d , use_sigmoid=True, gpu_ids=gpu_ids )
    netD3 = NLayerDiscriminator(image_nc*2, ndf, n_layers= 4, norm_layer=nn.BatchNorm2d, use_sigmoid=True, gpu_ids=gpu_ids)
    if len(gpu_ids) > 0:
        netD2.cuda()
        netD3.cuda()
    netD2.apply(weights_init)
    netD3.apply(weights_init)
    return netD2, netD3

class CA_NET1(nn.Module):

    def __init__(self, attrB, nt, gpu_ids=[]):
        super(CA_NET1, self).__init__()
        self.c_dim = nt
        self.fc1 = nn.Linear(attrB, self.c_dim * 2, bias=False)
        self.relu = nn.ReLU()
        self.gpu_ids = gpu_ids

    def encode(self, text_embedding):
        x = self.relu(self.fc1(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if len(self.gpu_ids) > 0:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

############# Networks for G1 #############
class G1(nn.Module):
    def __init__(self, sketch_nc, nz, nt, ngf, attrB,gpu_ids=[]):
        super(G1, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nt
        self.z_dim = nz
        self.define_module(attrB, nt, sketch_nc, gpu_ids)

    def define_module(self, attrB, nt, sketch_nc, gpu_ids):
        ninput = self.gf_dim * 16 + self.ef_dim
        ngf = self.gf_dim

        self.ca_net1 = CA_NET1(attrB, nt, gpu_ids)
        self.decoder = nn.Sequential(
            upBlock(ngf, ngf // 2),
            upBlock(ngf // 2, ngf // 4),
            upBlock(ngf // 4, ngf // 8),
            upBlock(ngf // 8, ngf // 16),
        )
        self.sp = nn.Sequential(
            conv3x3(ngf // 16, sketch_nc),
            nn.Tanh(),
        )
        self.cc = nn.Sequential(
            conv3x3(ngf // 16, sketch_nc),
        )
    def forward(self, l_code, z_code):
        l_code = l_code.view(-1, self.gf_dim, 4, 4)
        recon_img = self.decoder(l_code)
        cc = self.cc(recon_img)
        sp = self.sp(recon_img)

        z_code = z_code.view(-1, self.gf_dim, 4, 4)
        fake_img = self.decoder(z_code)
        fsp = self.sp(fake_img)
        fcc = self.cc(fake_img)
        return [sp, cc], [fsp, fcc]

class G2(nn.Module):
    def __init__(self, input_nc, output_nc, nz, nt, ngf, attr, gpu_ids=[]):
        super(G2, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nt
        self.output_nc = output_nc
        self.define_module(input_nc, output_nc, nt, attr, gpu_ids)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(2):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self, input_nc, output_nc, nt, attr, gpu_ids):
        ngf = self.gf_dim
        self.ca_net = CA_NET2(attr, nt, gpu_ids)

        self.encoder = nn.Sequential(
            conv3x3(input_nc, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )
        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + ngf * 8, ngf * 8),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        self.residual = self._make_layer(ResBlock, ngf * 8)
        self.upsample1 = upBlock(ngf * 8, ngf * 4)
        self.upsample2 = upBlock(ngf * 4, ngf * 2)
        self.upsample3 = upBlock(ngf * 2, ngf * 1)
        self.img = nn.Sequential(
            conv3x3(ngf * 1, self.output_nc),
            nn.Tanh()
        )

    def forward(self, stage1_img, text_embedding):
        encoded_img = self.encoder(stage1_img)
        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 8, 8)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        fake_img = self.img(h_code)
        return fake_img, mu, logvar

class ConUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, attr, ngf=64):
        super(ConUnetGenerator, self).__init__()
        self.ngf        = ngf
        self.attD 		= attr
        self.image_size = 64
        self.down1 = nn.Sequential(
            # nn.Dropout2d(0.7),
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1),
            # nn.BatchNorm2d(ngf),
            nn.InstanceNorm2d(ngf),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1),
            # nn.BatchNorm2d(ngf*2),
            nn.InstanceNorm2d(ngf*2),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down3 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1),
            # nn.BatchNorm2d(ngf*4),
            nn.InstanceNorm2d(ngf*4),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down4 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2,padding=1),
            # nn.BatchNorm2d(ngf*8),
            nn.InstanceNorm2d(ngf*8),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down5 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf*16),
            nn.InstanceNorm2d(ngf*16),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )


        self.hr_joint = nn.Sequential(
            conv3x3(attr+ngf*16, ngf*16),
            # nn.BatchNorm2d(ngf*4),
            nn.InstanceNorm2d(ngf*16),
            nn.ReLU(True),
        )
        self.residual = self._make_layer(ResBlock, ngf*16)

        self.unet = 2
        self.up5  = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16 * self.unet, ngf*8, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf*8),
            nn.InstanceNorm2d(ngf*8),
            nn.ReLU(True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * self.unet, ngf * 4, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf*4),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * self.unet, ngf * 2, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf*2),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * self.unet, ngf, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 1 * self.unet, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


        self.fg5 = nn.Linear(self.attD, 1)
        self.bg5  = nn.Linear(self.attD, ngf*16*2*2)

        self.fcf = nn.Linear(ngf*16*2*2, self.attD)
        self.fcb = nn.Linear(self.attD, ngf*16*2*2)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(2):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, input):
    # def forward(self, input, attr):
        m1 = self.down1(input)
        m2 = self.down2(m1)
        m3 = self.down3(m2)
        m4 = self.down4(m3)
        m5 = self.down5(m4)

        # gamma5 = self.fg5(attr.view(-1, self.attD)).view(-1,1,1,1).repeat(1,self.ngf*16,2,2)
        # beta5 = self.bg5(attr.view(-1, self.attD)).view(-1, self.ngf*16, 2, 2)
        # mf = gamma5 * m5 + beta5
        mf = m5

        mf  = mf.view(-1,self.ngf*16*2*2)
        mf1 = self.fcf(mf)
        mf  = self.fcb(mf1)
        mf  = mf.view(-1,self.ngf*16,2,2)

        # attr = attr.view(-1, self.attD, 1, 1)
        # attr = attr.repeat(1, 1, 2, 2)

        # mf = self.hr_joint(torch.cat([m5, attr], 1))
        # mf = self.residual(mf)

        if(self.unet==2):
            u5 = self.up5(torch.cat([mf, m5], 1))
            u4 = self.up4(torch.cat([u5, m4], 1))
            u3 = self.up3(torch.cat([u4, m3], 1))
            u2 = self.up2(torch.cat([u3, m2], 1))
            output = self.up1(torch.cat([u2, m1], 1))
        else:
            u5 = self.up5(mf)
            u4 = self.up4(u5)
            u3 = self.up3(u4)
            u2 = self.up2(u3)
            output = self.up1(u2)

        return output, mf1

class ConUnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, attr, ngf=64):
        super(ConUnetEncoder, self).__init__()
        self.ngf        = ngf
        self.attD 		= attr
        self.image_size = 64
        self.down1 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf),
            # nn.InstanceNorm2d(ngf),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf*2),
            # nn.InstanceNorm2d(ngf*2),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down3 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf*4),
            # nn.InstanceNorm2d(ngf*4),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down4 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf*8),
            # nn.InstanceNorm2d(ngf*8),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down5 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*16),
            # nn.InstanceNorm2d(ngf*16),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )

        self.fcf = nn.Linear(ngf*16*2*2, self.attD)

    def forward(self, input):

        m1 = self.down1(input)
        m2 = self.down2(m1)
        m3 = self.down3(m2)
        m4 = self.down4(m3)
        m5 = self.down5(m4)

        return m5, m4, m3, m2, m1

class ConUnetDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, attr, ngf=64, unet_flag = True):
        super(ConUnetDecoder, self).__init__()
        self.ngf        = ngf
        self.attD 		= attr
        self.image_size = 64
        self.down1 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf),
            # nn.InstanceNorm2d(ngf),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf*2),
            # nn.InstanceNorm2d(ngf*2),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down3 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf*4),
            # nn.InstanceNorm2d(ngf*4),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down4 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf*8),
            # nn.InstanceNorm2d(ngf*8),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down5 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*16),
            # nn.InstanceNorm2d(ngf*16),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )


        self.hr_joint = nn.Sequential(
            conv3x3(attr+ngf*16, ngf*16),
            nn.BatchNorm2d(ngf*16),
            # nn.InstanceNorm2d(ngf*16),
            nn.ReLU(True),
        )

        self.hr_joint4 = nn.Sequential(
            conv3x3(attr+ngf*8, ngf*8),
            nn.BatchNorm2d(ngf*8),
            # nn.InstanceNorm2d(ngf*16),
            nn.ReLU(True),
        )

        self.hr_joint3 = nn.Sequential(
            conv3x3(attr+ngf*4, ngf*4),
            nn.BatchNorm2d(ngf*4),
            # nn.InstanceNorm2d(ngf*16),
            nn.ReLU(True),
        )

        self.hr_joint2 = nn.Sequential(
            conv3x3(attr+ngf*2, ngf*2),
            nn.BatchNorm2d(ngf*2),
            # nn.InstanceNorm2d(ngf*16),
            nn.ReLU(True),
        )

        self.hr_joint1 = nn.Sequential(
            conv3x3(attr+ngf*1, ngf*1),
            nn.BatchNorm2d(ngf*1),
            # nn.InstanceNorm2d(ngf*16),
            nn.ReLU(True),
        )

        self.residual = self._make_layer(ResBlock, ngf*16)

        self.unet_flag = unet_flag
        if(self.unet_flag):
            self.unet=2
        else:
            self.unet=1
        self.up5  = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16 * self.unet, ngf*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*8),
            # nn.InstanceNorm2d(ngf*8),
            nn.ReLU(True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * self.unet, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4),
            # nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * self.unet, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            # nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * self.unet, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            # nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 1 * self.unet, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


        self.fg5  = nn.Linear(self.attD, 1)
        self.bg5  = nn.Linear(self.attD, ngf*16*2*2)

        # self.fg5_  = nn.Linear(self.attD, ngf*16*2*2)
        # self.bg5_  = nn.Linear(self.attD, ngf*16*2*2)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(2):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, m5, m4, m3, m2, m1, attr):

        # gamma5 = self.fg5(attr.view(-1, self.attD)).view(-1,1,1,1).repeat(1,self.ngf*16,2,2)
        # beta5 = self.bg5(attr.view(-1, self.attD)).view(-1, self.ngf*16, 2, 2)

        # gamma5 = self.fg5_(attr.view(-1, self.attD)).view(-1, self.ngf*16, 2, 2)
        # beta5 = self.bg5_(attr.view(-1, self.attD)).view(-1, self.ngf*16, 2, 2)

        # mf = gamma5 * m5 + beta5
        # mf = m5
        # pdb.set_trace()
        mf = self.hr_joint(torch.cat([m5, attr.repeat(1,1,2,2)], 1))
        m4 = self.hr_joint4(torch.cat([m4, attr.repeat(1,1,4,4)], 1))
        m3 = self.hr_joint3(torch.cat([m3, attr.repeat(1,1,8,8)], 1))
        m2 = self.hr_joint2(torch.cat([m2, attr.repeat(1,1,16,16)], 1))
        m1 = self.hr_joint1(torch.cat([m1, attr.repeat(1,1,32,32)], 1))

        if(self.unet==2):
            u5 = self.up5(torch.cat([mf, m5], 1))
            u4 = self.up4(torch.cat([u5, m4], 1))
            u3 = self.up3(torch.cat([u4, m3], 1))
            u2 = self.up2(torch.cat([u3, m2], 1))
            output = self.up1(torch.cat([u2, m1], 1))
        else:
            u5 = self.up5(mf)
            u4 = self.up4(u5)
            u3 = self.up3(u4)
            u2 = self.up2(u3)
            output = self.up1(u2)

        return output

class ConUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, attr, ngf=64):
        super(ConUnetGenerator, self).__init__()
        self.ngf        = ngf
        self.attD 		= attr
        self.image_size = 64
        self.down1 = nn.Sequential(
            # nn.Dropout2d(0.7),
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1),
            # nn.BatchNorm2d(ngf),
            nn.InstanceNorm2d(ngf),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1),
            # nn.BatchNorm2d(ngf*2),
            nn.InstanceNorm2d(ngf*2),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down3 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1),
            # nn.BatchNorm2d(ngf*4),
            nn.InstanceNorm2d(ngf*4),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down4 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2,padding=1),
            # nn.BatchNorm2d(ngf*8),
            nn.InstanceNorm2d(ngf*8),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )
        self.down5 = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf*16),
            nn.InstanceNorm2d(ngf*16),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, True),
        )


        self.hr_joint = nn.Sequential(
            conv3x3(attr+ngf*16, ngf*16),
            # nn.BatchNorm2d(ngf*4),
            nn.InstanceNorm2d(ngf*16),
            nn.ReLU(True),
        )
        self.residual = self._make_layer(ResBlock, ngf*16)

        self.unet = 1
        self.up5  = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16 * self.unet, ngf*8, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf*8),
            nn.InstanceNorm2d(ngf*8),
            nn.ReLU(True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * self.unet, ngf * 4, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf*4),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * self.unet, ngf * 2, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf*2),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * self.unet, ngf, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 1 * self.unet, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


        self.fg5 = nn.Linear(self.attD, 1)
        self.bg5  = nn.Linear(self.attD, ngf*16*2*2)

        self.fcf = nn.Linear(ngf*16*2*2, self.attD)
        self.fcb = nn.Linear(self.attD, ngf*16*2*2)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(2):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, input):
    # def forward(self, input, attr):
        m1 = self.down1(input)
        m2 = self.down2(m1)
        m3 = self.down3(m2)
        m4 = self.down4(m3)
        m5 = self.down5(m4)

        # gamma5 = self.fg5(attr.view(-1, self.attD)).view(-1,1,1,1).repeat(1,self.ngf*16,2,2)
        # beta5 = self.bg5(attr.view(-1, self.attD)).view(-1, self.ngf*16, 2, 2)
        # mf = gamma5 * m5 + beta5
        mf = m5

        mf  = mf.view(-1,self.ngf*16*2*2)
        mf1 = self.fcf(mf)
        mf  = self.fcb(mf1)
        mf  = mf.view(-1,self.ngf*16,2,2)

        # attr = attr.view(-1, self.attD, 1, 1)
        # attr = attr.repeat(1, 1, 2, 2)

        # mf = self.hr_joint(torch.cat([m5, attr], 1))
        # mf = self.residual(mf)

        if(self.unet==2):
            u5 = self.up5(torch.cat([mf, m5], 1))
            u4 = self.up4(torch.cat([u5, m4], 1))
            u3 = self.up3(torch.cat([u4, m3], 1))
            u2 = self.up2(torch.cat([u3, m2], 1))
            output = self.up1(torch.cat([u2, m1], 1))
        else:
            u5 = self.up5(mf)
            u4 = self.up4(u5)
            u3 = self.up3(u4)
            u2 = self.up2(u3)
            output = self.up1(u2)

        return output, mf1

class ConUnetEncoderToy(nn.Module):
    def __init__(self, input_nc, output_nc, attr, ngf=64):
        super(ConUnetEncoderToy, self).__init__()
        self.ngf        = ngf
        self.attD 		= attr
        self.image_size = 64
        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True),
        )

        self.fcf = nn.Linear(ngf*4*8*8, 2)

    def forward(self, input):
        m1 = self.down1(input)
        m2 = self.down2(m1)
        m3 = self.down3(m2)
        m4 = self.fcf(m3.view(-1,self.ngf*4*8*8))

        return m4

class ConUnetDecoderToy(nn.Module):
    def __init__(self, input_nc, output_nc, attr, ngf=64):
        super(ConUnetDecoderToy, self).__init__()
        self.ngf        = ngf
        self.attD 		= attr
        self.image_size = 64

        self.unet = 1
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * self.unet, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * self.unet, ngf, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 1 * self.unet, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


        self.fg5 = nn.Linear(self.attD, 1)
        self.bg5  = nn.Linear(self.attD, 2)

        self.fcb = nn.Linear(2, ngf*4*8*8)

    def forward(self, m4, attr):

        gamma5 = self.fg5(attr.view(-1, self.attD)).repeat(1,2)
        beta5 = self.bg5(attr.view(-1, self.attD)).view(-1, 2)
        mf = gamma5 * m4 + beta5

        # mf = m4

        mf = self.fcb(mf).view(-1, self.ngf*4, 8, 8)

        u3 = self.up3(mf)
        u2 = self.up2(u3)
        output = self.up1(u2)

        return output

class AutoToy(nn.Module):
    def __init__(self, input_nc, output_nc, attr, ngf=64):
        super(AutoToy, self).__init__()
        self.ngf        = ngf
        self.attD 		= attr
        self.image_size = 64

        self.unet = 1
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * self.unet, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * self.unet, ngf, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 1 * self.unet, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True),
        )

        self.fcf = nn.Linear(ngf*4*8*8, 2)
        self.fcb = nn.Linear(2, ngf*4*8*8)

    def forward(self, input):

        m1 = self.down1(input)
        m2 = self.down2(m1)
        m3 = self.down3(m2)
        m4 = self.fcf(m3.view(-1,self.ngf*4*8*8))

        mf = m4

        mf = self.fcb(mf)

        u3 = self.up3(u4)
        u2 = self.up2(u3)
        output = self.up1(u2)

        return output

class FeatureClassifierToy(nn.Module):
    def __init__(self, ngf, no_closed):
        super(FeatureClassifierToy, self).__init__()
        self.ngf = ngf
        self.fc1 = nn.Linear( 2, no_closed)

    def forward(self, input):
        xf  = input
        out = self.fc1(xf)
        return out

class ConGeneratorAuto(nn.Module):
    def __init__(self, input_nc, output_nc, attr, ngf=64, nlyr=2, nrm='batch', unet=False):
        super(ConGeneratorAuto, self).__init__()
        self.ngf        = ngf
        self.attD 		= attr
        self.image_size = 64
        self.nlyr 		= nlyr
        self.nrm 		= nrm
        self.h 			= int(self.image_size/np.power(2,(self.nlyr)))
        self.unet 		= unet
        mult 	 		= 1
        if(self.unet==True):
            mult = 2

        if(self.nrm=='batch'):
            norm_layer = nn.BatchNorm2d
        elif(self.nrm=='instance'):
            norm_layer = nn.InstanceNorm2d

        self.down_seq = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1)]
        if (not (self.nrm=='none')):
            self.down_seq+= [norm_layer(ngf)]
        else:
            pass
        self.down_seq+= [nn.LeakyReLU(0.2, True)]

        for i in range(0, self.nlyr-1):
            self.down_seq+= [nn.Conv2d(ngf * np.power(2,i), ngf * np.power(2,i+1), kernel_size=4, stride=2, padding=1)]
            if (not (self.nrm=='none')):
                self.down_seq+= [norm_layer(ngf * np.power(2,i+1))]
            else:
                pass
            self.down_seq+= [nn.LeakyReLU(0.2, True)]

        self.up_seq = [nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)]
        for i in range(self.nlyr-1, 0, -1):
            if(i==(self.nlyr-1)):
                self.up_seq = [nn.ConvTranspose2d(ngf * np.power(2,i) * mult, ngf * np.power(2,i-1), kernel_size=4, stride=2, padding=1)]
            else:
                self.up_seq+= [nn.ConvTranspose2d(ngf * np.power(2,i) * mult, ngf * np.power(2,i-1), kernel_size=4, stride=2, padding=1)]
            if (not (self.nrm=='none')):
                self.up_seq+= [norm_layer(ngf * np.power(2,i-1))]
            else:
                pass
            self.up_seq+= [nn.LeakyReLU(0.2, True)]

        self.up_seq+= [nn.ConvTranspose2d(ngf * 1 * mult, output_nc, kernel_size=4, stride=2, padding=1),]
        self.up_seq+= [nn.Tanh()]

        self.down = nn.Sequential(*self.down_seq)
        self.up   = nn.Sequential(*self.up_seq)

        self.fg = nn.Linear(self.attD, 1)
        self.bg = nn.Linear(self.attD, self.ngf * self.nlyr * 2 * self.h * self.h)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(2):
            layers.append(block(channel_num))
        return layers

    def forward(self, input, attr):

        feat = input
        cat_list = {}
        if(self.unet):
            for i in range(self.nlyr):
                feat = self.down[i](feat)
                cat_list[str(i)] = feat
        else:
            feat = self.down(input)

        gamma = self.fg(attr.view(-1, self.attD)).view(-1,1,1,1).repeat(1, self.ngf * self.nlyr * 2, self.h, self.h)
        beta = self.bg(attr.view(-1, self.attD)).view(-1, self.ngf * self.nlyr * 2, self.h, self.h)
        feat_cond = gamma * feat + beta

        if(self.unet):
            new_in   = feat_cond
            for i in range(self.nlyr, 0, -1):
                new_in = torch.cat([new_in, cat_list[str(i)]])
            output = new_in
        else:
            output = self.up(feat_cond)

        return output, feat

class FeatureClassifier(nn.Module):
    def __init__(self, ngf, no_closed):
        super(FeatureClassifier, self).__init__()
        self.ngf = ngf
        self.fc1 = nn.Linear( ngf*16*2*2, no_closed)

    def forward(self, input):
        xf  = input.view(-1, self.ngf*16*2*2)
        out = self.fc1(xf)
        return out

class ConfidenceEstimator(nn.Module):
    def __init__(self, ngf):
        super(ConfidenceEstimator, self).__init__()
        self.ngf = ngf
        self.fc1 = nn.Linear( ngf*16*2*2, 1)

    def forward(self, input):
        xf  = input.view(-1, self.ngf*16*2*2)
        out = self.fc1(xf)
        return out

class DomainClassifier(nn.Module):
    def __init__(self, ngf):
        super(DomainClassifier, self).__init__()
        self.ngf = ngf
        self.fc1 = nn.Linear( ngf*16*2*2, 2)

    def forward(self, input):
        xf = input.view(-1, self.ngf*16*2*2)
        out = self.fc1(xf)
        return out

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        x = self.model(input)
        # x = torch.mean(x.view(-1,x.size()[1]*x.size()[2]*x.size()[3]),1)
        x = x.view(-1,x.size()[1]*x.size()[2]*x.size()[3])
        return x



class encoder32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        # Shortcut out of the network at 8x8
        self.conv_out_6 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        # Shortcut out of the network at 4x4
        self.conv_out_9 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv10 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        self.conv_out_10 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(128)

        # self.bn4 = nn.BatchNorm2d(128)
        # self.bn5 = nn.BatchNorm2d(128)
        # self.bn6 = nn.BatchNorm2d(128)

        # self.bn7 = nn.BatchNorm2d(128)
        # self.bn8 = nn.BatchNorm2d(128)
        # self.bn9 = nn.BatchNorm2d(128)
        # self.bn10 = nn.BatchNorm2d(128)

        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(64)
        self.bn3 = nn.InstanceNorm2d(128)

        self.bn4 = nn.InstanceNorm2d(128)
        self.bn5 = nn.InstanceNorm2d(128)
        self.bn6 = nn.InstanceNorm2d(128)

        self.bn7 = nn.InstanceNorm2d(128)
        self.bn8 = nn.InstanceNorm2d(128)
        self.bn9 = nn.InstanceNorm2d(128)
        self.bn10 = nn.InstanceNorm2d(128)

        self.fc1 = nn.Linear(128*2*2, latent_size)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.dr4 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        # self.cuda()

    def forward(self, x, output_scale=1):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 8 x 8
        if output_scale == 8:
            x = self.conv_out_6(x)
            x = x.view(batch_size, -1)
            # x = clamp_to_unit_sphere(x, 8*8)
            return x

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 4x4
        if output_scale == 4:
            x = self.conv_out_9(x)
            x = x.view(batch_size, -1)
            # x = clamp_to_unit_sphere(x, 4*4)
            return x

        x = self.dr4(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 2x2
        if output_scale == 2:
            x = self.conv_out_10(x)
            x = x.view(batch_size, -1)
            # x = clamp_to_unit_sphere(x, 2*2)
            return x

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = clamp_to_unit_sphere(x)
        return x

class generator32(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, n=6, ac_scale=4, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False)
        self.n = n
        self.ac_scale = ac_scale
        self.mul = nn.Linear(n, 1)
        self.add  = nn.Linear(n, self.ac_scale*4*self.latent_size)

        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False)
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False)
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1)

        self.bn1 = nn.InstanceNorm2d(512)
        self.bn2 = nn.InstanceNorm2d(512)
        self.bn3 = nn.InstanceNorm2d(256)
        self.bn4 = nn.InstanceNorm2d(128)

        self.batch_size = batch_size
        self.apply(weights_init)
        # self.cuda()


    def forward(self, x, attr):
        batch_size = x.shape[0]

        gamma = self.mul(attr.view(-1, self.n)).view(-1,1).repeat(1, self.ac_scale*4*self.latent_size)
        beta = self.add(attr.view(-1, self.n)).view(-1, self.ac_scale*4*self.latent_size)

        x = gamma * x + beta

        if self.ac_scale <= 1:

            x = self.fc1(x)
            x = x.resize(batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if self.ac_scale == 2:
            x = x.view(batch_size, self.latent_size, 2, 2)
            x = self.conv2_in(x)
        if self.ac_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            x = self.bn2(x)

        # 512 x 4 x 4
        if self.ac_scale == 4:

            x = x.view(batch_size, self.latent_size, 4, 4)
            x = self.conv3_in(x)
        if self.ac_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            x = self.bn3(x)

        # 256 x 8 x 8
        if self.ac_scale == 8:
            x = x.view(batch_size, self.latent_size, 8, 8)
            x = self.conv4_in(x)
        if self.ac_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            x = self.bn4(x)
        # 128 x 16 x 16
        x = self.conv5(x)
        # 3 x 32 x 32
        # x = nn.Sigmoid()(x)
        x = nn.Tanh()(x)
        return x

class classifierM(nn.Module):
    def __init__(self, latent_size=100, no_closed=1):
        super(classifierM, self).__init__()
        self.fc1 = nn.Linear( latent_size, no_closed)

    def forward(self, input):
        # xf  = input.view(-1, self.ngf*16*2*2)
        out = self.fc1(input)
        return out

class encoderM(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        # Shortcut out of the network at 8x8
        self.conv_out_6 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        # Shortcut out of the network at 4x4
        self.conv_out_9 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv10 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        self.conv_out_10 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(128)

        # self.bn4 = nn.BatchNorm2d(128)
        # self.bn5 = nn.BatchNorm2d(128)
        # self.bn6 = nn.BatchNorm2d(128)

        # self.bn7 = nn.BatchNorm2d(128)
        # self.bn8 = nn.BatchNorm2d(128)
        # self.bn9 = nn.BatchNorm2d(128)
        # self.bn10 = nn.BatchNorm2d(128)

        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(64)
        self.bn3 = nn.InstanceNorm2d(128)

        self.bn4 = nn.InstanceNorm2d(128)
        self.bn5 = nn.InstanceNorm2d(128)
        self.bn6 = nn.InstanceNorm2d(128)

        self.bn7 = nn.InstanceNorm2d(128)
        self.bn8 = nn.InstanceNorm2d(128)
        self.bn9 = nn.InstanceNorm2d(128)
        self.bn10 = nn.InstanceNorm2d(128)

        self.fc1 = nn.Linear(128*2*2, latent_size)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.dr4 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        # self.cuda()

    def forward(self, x, output_scale=1):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 8 x 8
        if output_scale == 8:
            x = self.conv_out_6(x)
            x = x.view(batch_size, -1)
            # x = clamp_to_unit_sphere(x, 8*8)
            return x

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 4x4
        if output_scale == 4:
            x = self.conv_out_9(x)
            x = x.view(batch_size, -1)
            x = clamp_to_unit_sphere(x, 4*4)
            return x

        x = self.dr4(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 2x2
        if output_scale == 2:
            x = self.conv_out_10(x)
            x = x.view(batch_size, -1)
            # x = clamp_to_unit_sphere(x, 2*2)
            return x

        x = x.view(batch_size, -1)
        x = self.fc1(x)

        # x = clamp_to_unit_sphere(x)
        return x

class generatorM(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, n=6, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False)
        self.n = n
        self.mul = nn.Linear(n, 1)
        self.add  = nn.Linear(n, 16*self.latent_size)

        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False)
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False)
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1)

        # self.bn1 = nn.BatchNorm2d(512)
        # self.bn2 = nn.BatchNorm2d(512)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.bn4 = nn.BatchNorm2d(128)

        self.bn1 = nn.InstanceNorm2d(512)
        self.bn2 = nn.InstanceNorm2d(512)
        self.bn3 = nn.InstanceNorm2d(256)
        self.bn4 = nn.InstanceNorm2d(128)

        self.batch_size = batch_size
        self.apply(weights_init)
        # self.cuda()


    def forward(self, x, input_scale=1):

        if input_scale <= 1:

            x = self.fc1(x)
            x = x.resize(self.batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if input_scale == 2:
            x = x.view(self.batch_size, self.latent_size, 2, 2)
            x = self.conv2_in(x)
        if input_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            x = self.bn2(x)

        # 512 x 4 x 4
        if input_scale == 4:

            x = x.view(self.batch_size, self.latent_size, 4, 4)
            x = self.conv3_in(x)
        if input_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            x = self.bn3(x)

        # 256 x 8 x 8
        if input_scale == 8:
            x = x.view(self.batch_size, self.latent_size, 8, 8)
            x = self.conv4_in(x)
        if input_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            x = self.bn4(x)
        # 128 x 16 x 16
        x = self.conv5(x)
        # 3 x 32 x 32
        # x = nn.Sigmoid()(x)
        x = nn.Tanh()(x)
        return x

class discriminator32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*4*4 * 2, 1)
        # self.fc1 = nn.Linear(128*4*4 * 2, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        # self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = x.view(batch_size, -1)
        if return_features:
            return x

        # Lazy minibatch discrimination: avg of other examples' features
        batch_avg = torch.exp(-x.mean(dim=0))
        batch_avg = batch_avg.expand(batch_size, -1)
        x = torch.cat([x, batch_avg], dim=1)
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        return x

def calc_gradient_penalty(netD, real_data, fake_data, penalty_lambda=10.0):
    alpha = torch.rand(real_data.size()[0], 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    # Traditional WGAN-GP
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    # Possibly more reasonable
    #interpolates = torch.cat([real_data, fake_data])
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    ones = torch.ones(disc_interpolates.size()).cuda()
    gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_lambda
    return penalty


## implementation of densenet cifar taken from bamos@github
class BottleneckDense(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(BottleneckDense, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayerDense(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayerDense, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class TransitionDense(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(TransitionDense, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet10(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet10, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = TransitionDense(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = TransitionDense(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(BottleneckDense(nChannels, growthRate))
            else:
                layers.append(SingleLayerDense(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out = F.log_softmax(self.fc(out))
        return out

class DenseClassifier10(nn.Module):
    def __init__(self, latent_size=342, no_closed=10):
        super(DenseClassifier10, self).__init__()
        self.fc1 = nn.Linear(latent_size, no_closed)

    def forward(self, input):
        # xf  = input.view(-1, self.ngf*16*2*2)
        out = self.fc1(input)
        return out




class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

## this following code is from  Jun-Yan Zhu and Taesung Park https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            # self.loss = nn.MSELoss()
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class Dense(nn.Module):
    def __init__(self, nt, gpu_ids=[]):
        super(Dense, self).__init__()
        self.ef_dim = nt
        self.relu = nn.ReLU(inplace=True)
        ############# 64x64  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.norm0 = haze_class.features.norm0
        self.relu0 = haze_class.features.relu0
        self.pool0 = haze_class.features.pool0

        ############# Block1-down 16x16  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 8x8  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  4x4 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        ############# Block4-up  512x2x2  ##############
        self.hr_joint = nn.Sequential(
            conv3x3(nt+512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.residual = self._make_layer(ResBlock, 512)

        self.bottle_block4 = BottleneckBlock(512, 256)
        self.trans_block4 = TransitionBlock(768, 128)

        ############# Block5-up  4x4 ##############
        self.bottle_block5 = BottleneckBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)

        ############# Block6-up 8x8   ##############
        self.bottle_block6 = BottleneckBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)

        ############# Block7-up 16x16   ##############
        self.bottle_block7 = BottleneckBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)

        ## 128 X  128
        ############# Block8-up c 32x32  ##############
        self.bottle_block8 = BottleneckBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.tanh = nn.Tanh()

        def _make_layer(self, block, channel_num):
            layers = []
            for i in range(2):
                layers.append(block(channel_num))
            return nn.Sequential(*layers)

        def forward(self, x, attr):
            ## 256x256
            x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))
            ## 64 X 64
            x1 = self.dense_block1(x0)
            x1 = self.trans_block1(x1)
            ###  32x32
            x2 = self.trans_block2(self.dense_block2(x1))
            ### 16 X 16
            x3 = self.trans_block3(self.dense_block3(x2))
            attr = attr.view(-1, self.ef_dim,1,1)
            attr = attr.repeat(1,1,2,2)
            xf = self.hr_joint(torch.cat([x3, attr],1))
            xf = self.residual(xf)
            ## 8 X 8
            x4 = self.trans_block4(self.bottle_block4(xf))
            x42 = torch.cat([x4, x2], 1)
            ## 16 X 16
            x5 = self.trans_block5(self.bottle_block5(x42))
            x52 = torch.cat([x5, x1], 1)
            ##  32 X 32
            x6 = self.trans_block6(self.bottle_block6(x52))
            ##  64 X 64
            x7 = self.trans_block7(self.bottle_block7(x6))
            ##  128 X 128
            x8 = self.trans_block8(self.bottle_block8(x7))
            x8 = torch.cat([x8, x], 1)
            x9 = self.relu(self.conv_refin(x8))
            out = self.tanh(self.refine3(x9))

            return out

################################### Building Blocks for Networks
## this following code is from Han Zhang's work: https://github.com/hanzhanggit/StackGAN-Pytorch
# Commonly used convolutional block
def clamp_to_unit_sphere(x, components=1):
    # If components=4, then we normalize each quarter of x independently
    # Useful for the latent spaces of fully-convolutional networks
    batch_size, latent_size = x.shape
    latent_subspaces = []
    for i in range(components):
        step = latent_size // components
        left, right = step * i, step * (i+1)
        subspace = x[:, left:right].clone()
        norm = torch.norm(subspace, p=2, dim=1)
        subspace = subspace / (norm.expand(1, -1).t()   + 1e-34)
        latent_subspaces.append(subspace)
    # Join the normalized pieces back together
    return torch.cat(latent_subspaces, dim=1)

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True)
    )
    return block

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

# Transition Block
class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

# Bottleneck Block
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

################################### Network Utility Functions
# weight initialization code
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

# printing network parameter details
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)






########################### toys
dimt=5
class e_toy(nn.Module):
    def __init__(self):
        super(e_toy, self).__init__()
        self.down = nn.Sequential(
            nn.Linear(2, dimt),
            # nn.LeakyReLU(0.2, True),
            nn.Sigmoid(),
            nn.Linear(dimt, dimt),
            # nn.LeakyReLU(0.2, True),
            nn.Sigmoid(),
            nn.Linear(dimt, dimt),
            # nn.LeakyReLU(0.2, True),
            nn.Sigmoid(),
            nn.Linear(dimt, dimt)
        )

    def forward(self, input):
        out = self.down(input)
        return out

class g_toy(nn.Module):
    def __init__(self, attr):
        super(g_toy, self).__init__()
        self.attr = attr
        self.up = nn.Sequential(
            nn.Linear(dimt, dimt),
            # nn.LeakyReLU(0.2, True),
            nn.Sigmoid(),
            nn.Linear(dimt, dimt),
            # nn.LeakyReLU(0.2, True),
            nn.Sigmoid(),
            nn.Linear(dimt, 2)
        )
        self.mul = nn.Linear(attr, 1)
        self.add  = nn.Linear(attr, dimt)

    def forward(self, input, a, use_cond):
        x = input
        if(use_cond):
            gamma = self.mul(a.view(-1, self.attr)).view(-1,1).repeat(1, dimt)
            beta = self.add(a.view(-1, self.attr)).view(-1, dimt)

            x = gamma * x + beta

        out = self.up(x)

        return out

class c_toy(nn.Module):
    def __init__(self, attr):
        super(c_toy, self).__init__()
        self.attr = attr
        self.classifiy = nn.Sequential(
            nn.Linear(dimt, dimt),
            # nn.LeakyReLU(0.2, True),
            nn.Sigmoid(),
            nn.Linear(dimt, attr)
        )

    def forward(self, input):
        out = self.classifiy(input)

        return out







############################# old network

class DCCA_Encoder(nn.Module):											# Archietecture from DRCN (Deep Reconstruction based Classification Network) ECCV 2016
    def __init__(self):
        super(DCCA_Encoder, self).__init__()
        self.no_class = 15
        self.encoder = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(3, 32, 3, stride=2, padding=(2,2)),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, 3, stride=2, padding=(2,2)),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, 3, stride=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.fc = nn.Linear(8*8*128,self.no_class)

    def forward(self, x):
        out = x
        out = self.encoder(out)
        out = out.view(-1,8*8*128)
        out = self.fc(out)
        return out

class DCCA_Decoder(nn.Module):											# Archietecture from DRCN (Deep Reconstruction based Classification Network) ECCV 2016
    def __init__(self):
        super(DCCA_Decoder, self).__init__()
        self.no_class = 15
        self.decoder = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.Dropout2d(0.5),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=(1,1),  output_padding=(0,0)),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.Dropout2d(0.5),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=(2,2), output_padding=(1,1)),
            nn.Tanh()
        )
        self.fc = nn.Linear(self.no_class,128*8*8)
        # self.bn = nn.BatchNorm1d(128*3*3)

    def forward(self, x):
        out = x
        out = self.fc(out)
        # out = self.bn(out)
        out = out.view(-1,128,8,8)
        out = self.decoder(out)
        return out

class DCCA_Label_Classifier(nn.Module):											# Label Classifier architecture definition for DRCN
    def __init__(self):												# (if you use this comment fc_out layer in the encoder-decoder models)
        super(DCCA_Label_Classifier, self).__init__()
        self.no_class = 15
        self.clf = nn.Sequential(
            # nn.ReLU(True),
            nn.Linear(self.no_class, self.no_class)
        )

    def forward(self, x):
        out = self.clf(x)
        return out









