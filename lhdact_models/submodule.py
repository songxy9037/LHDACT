from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import Softmax
from torch.nn.parameter import Parameter

import models
from models.Transformer_ import TR
from models import ECA as raa
from models import FPT as F1
from models import do_conv_pytorch as doconv
from einops import rearrange

from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d





def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    '''
    DO-Conv无痛涨点：使用over-parameterized卷积层提高CNN性能
    对于输入特征，先使用权重进行depthwise卷积，对输出结果进行权重为的传统卷积，
    '''
    return nn.Sequential(doconv.DOConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

def conv2(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))



class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class LHDACTbone(nn.Module):
    def __init__(self):
        super(LHDACTbone, self).__init__()
        self.inplanes = 32
        inplanes1 = 64
        k_size = 3
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 4)
        #############DAA
        self.conv5a = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.sa = PAM_Module(inplanes1)
        # self.sc = DA.CAM_Module(inplanes//2)
        self.ca = CAM_Module(inplanes1)
        ##############RAA
        self.raa = raa.eca_layer(inplanes1, k_size)
        ##

        self.conv51 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.convgra = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inplanes1),
                                     nn.ReLU())
        self.convsum = nn.Sequential(nn.Conv2d(inplanes1 * 4, inplanes1, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inplanes1),
                                     nn.ReLU())
        # self.conv53 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inplanes//2, inplanes*2, 1))
        self.conv7 = nn.Conv2d(16, 64, kernel_size=1, padding=0, stride=1, bias=False)
        #####################
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.fpt = F1.FPT(32)
        self.lastconv = nn.Sequential(convbn(384, 64, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))
        self.lastconv1 = nn.Sequential(convbn(384, 128, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 12, kernel_size=1, padding=0, stride=1, bias=False))
        self.graph = pvig_b_224_gelu()
        self.hilo = HiLo(inplanes1)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        ################DANet
        # PAM
        feat1 = self.conv5a(output_skip)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        #先降维 CAM
        feat2 = self.conv5c(output_skip)
        sc_feat = self.ca(feat2)
        sc_conv = self.conv52(sc_feat)
        #sc_output = self.conv7(sc_conv)
        feat_sum = sa_conv+sc_conv

        ####################

        output_branch1 = self.branch1(feat_sum)
        # output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(feat_sum)
        # output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(feat_sum)
        # output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(feat_sum)
        # output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        # output_branch1, output_branch2, output_branch3, output_branch4 = self.fpt(output_branch1, output_branch2,
        #                                                                           output_branch3, output_branch4)
        # 金字塔上采样
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        gwc_feature = torch.cat((output_raw, output_skip, feat_sum, output_branch4, output_branch3,
                                 output_branch2, output_branch1), 1)
        # print(output_branch1.shape)
        # feature1 = self.lastconv(output_branch1)
        # feature2 = self.lastconv(output_branch2)
        # feature3 = self.lastconv(output_branch3)
        # feature4 = self.lastconv(output_branch4)
        concat_feature = self.lastconv(gwc_feature)

        return concat_feature

class feature_extraction_nocat(nn.Module):
    def __init__(self):
        super(feature_extraction_nocat, self).__init__()
        self.inplanes = 32
        inplanes1 = 64
        k_size = 3
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.firstconv_gra = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(32, 128, 3, 2, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(128, 128, 3, 1, 1, 1),
                                           nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 4)
        #############DAA
        self.conv5a = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.sa = PAM_Module(inplanes1)
        # self.sc = DA.CAM_Module(inplanes//2)

        ##############RAA
        self.raa = raa.eca_layer(inplanes1, k_size)
        ##

        self.conv51 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.convgra = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inplanes1),
                                     nn.ReLU())
        self.convsum = nn.Sequential(nn.Conv2d(inplanes1 * 4, inplanes1, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inplanes1),
                                     nn.ReLU())
        # self.conv53 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inplanes//2, inplanes*2, 1))
        self.conv7 = nn.Conv2d(16, 64, kernel_size=1, padding=0, stride=1, bias=False)
        #####################
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.fpt = F1.FPT(32)
        self.lastconv = nn.Sequential(convbn(448, 64, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))
        self.lastconv1 = nn.Sequential(convbn(384, 128, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 12, kernel_size=1, padding=0, stride=1, bias=False))
        self.graph = pvig_b_224_gelu()
        self.hilo = HiLo(inplanes1)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        ###########Graph
        # out_gra = self.firstconv_gra(x)
        output_grap = self.graph(output_skip)
        output_grap = self.convgra(output_grap)
        # output_sum = torch.cat((output_skip, output_grap), dim=1)
        # output_sum = self.convsum(output_grap)

        ############HiLo
        output_line = rearrange(output_grap, 'b c h w -> b (h w) c')
        out_hilo = self.hilo(output_line, 64, 64)
        # print(out_hilo.shape)
        ################DANet
        # PAM
        # feat1 = self.conv5a(output_sum)
        # sa_feat = self.sa(feat1)
        # sa_conv = self.conv51(sa_feat)
        # #先降维 CAM
        # feat2 = self.conv5c(output_sum)
        # sc_feat = self.raa(feat2)
        # sc_conv = self.conv52(sc_feat)
        # #sc_output = self.conv7(sc_conv)
        # feat_sum = sa_conv+sc_conv
        # output_line = rearrange(output_sum, 'b c h w -> b (h w) c')
        # feat_sum_hilo = self.hilo(output_line, 64, 64)
        # print("DA",feat_sum.size())
        ####################

        output_branch1 = self.branch1(out_hilo)
        # output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(out_hilo)
        # output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(out_hilo)
        # output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(out_hilo)
        # output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        output_branch1, output_branch2, output_branch3, output_branch4 = self.fpt(output_branch1, output_branch2,
                                                                                  output_branch3, output_branch4)
        # 金字塔上采样
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        gwc_feature = torch.cat((output_grap, out_hilo, output_raw, output_skip, output_branch4, output_branch3,
                                 output_branch2, output_branch1), 1)
        # print(output_branch1.shape)
        # feature1 = self.lastconv(output_branch1)
        # feature2 = self.lastconv(output_branch2)
        # feature3 = self.lastconv(output_branch3)
        # feature4 = self.lastconv(output_branch4)
        concat_feature = self.lastconv(gwc_feature)

        return concat_feature


class feature_extraction_nomsp(nn.Module):
    def __init__(self):
        super(feature_extraction_nomsp, self).__init__()
        self.inplanes = 32
        inplanes1 = 64
        k_size = 3
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.firstconv_gra = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(32, 128, 3, 2, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(128, 128, 3, 1, 1, 1),
                                           nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 4)
        #############DAA
        self.conv5a = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.sa = PAM_Module(inplanes1)
        # self.sc = DA.CAM_Module(inplanes//2)

        ##############RAA
        self.raa = raa.eca_layer(inplanes1, k_size)
        ##

        self.conv51 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.convgra = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inplanes1),
                                     nn.ReLU())
        self.convsum = nn.Sequential(nn.Conv2d(inplanes1 * 4, inplanes1, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inplanes1),
                                     nn.ReLU())
        # self.conv53 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inplanes//2, inplanes*2, 1))
        self.conv7 = nn.Conv2d(16, 64, kernel_size=1, padding=0, stride=1, bias=False)
        #####################
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.fpt = F1.FPT(32)
        self.lastconv = nn.Sequential(convbn(320, 64, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))
        self.lastconv1 = nn.Sequential(convbn(384, 128, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 12, kernel_size=1, padding=0, stride=1, bias=False))
        self.graph = pvig_b_224_gelu()
        self.hilo = HiLo(inplanes1)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        ###########Graph
        out_gra = self.firstconv_gra(x)
        output_grap = self.graph(out_gra)
        # output_grap = self.convgra(output_grap)
        output_sum = torch.cat((output_skip, output_grap), dim=1)
        output_sum = self.convsum(output_sum)

        ############HiLo
        output_line = rearrange(output_sum, 'b c h w -> b (h w) c')
        out_hilo = self.hilo(output_line, 64, 64)
        # print(out_hilo.shape)
        ################DANet
        # PAM
        # feat1 = self.conv5a(output_sum)
        # sa_feat = self.sa(feat1)
        # sa_conv = self.conv51(sa_feat)
        # #先降维 CAM
        # feat2 = self.conv5c(output_sum)
        # sc_feat = self.raa(feat2)
        # sc_conv = self.conv52(sc_feat)
        # #sc_output = self.conv7(sc_conv)
        # feat_sum = sa_conv+sc_conv
        # output_line = rearrange(output_sum, 'b c h w -> b (h w) c')
        # feat_sum_hilo = self.hilo(output_line, 64, 64)
        # print("DA",feat_sum.size())
        ####################

        # output_branch1 = self.branch1(out_hilo)
        # # output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        #
        # output_branch2 = self.branch2(out_hilo)
        # # output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        #
        # output_branch3 = self.branch3(out_hilo)
        # # output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        #
        # output_branch4 = self.branch4(out_hilo)
        # # output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        # output_branch1, output_branch2, output_branch3, output_branch4 = self.fpt(output_branch1, output_branch2,
        #                                                                           output_branch3, output_branch4)
        # # 金字塔上采样
        # output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        # output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        # output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        # output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        gwc_feature = torch.cat((output_sum, out_hilo, output_raw, output_skip), 1)
        # print(output_branch1.shape)
        # feature1 = self.lastconv(output_branch1)
        # feature2 = self.lastconv(output_branch2)
        # feature3 = self.lastconv(output_branch3)
        # feature4 = self.lastconv(output_branch4)
        concat_feature = self.lastconv(gwc_feature)

        return concat_feature

class feature_extraction_nofpt(nn.Module):
    def __init__(self):
        super(feature_extraction_nofpt, self).__init__()
        self.inplanes = 32
        inplanes1 = 64
        k_size = 3
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.firstconv_gra = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(32, 128, 3, 2, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(128, 128, 3, 1, 1, 1),
                                           nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 4)
        #############DAA
        self.conv5a = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.sa = PAM_Module(inplanes1)
        # self.sc = DA.CAM_Module(inplanes//2)

        ##############RAA
        self.raa = raa.eca_layer(inplanes1, k_size)
        ##

        self.conv51 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes1),
                                    nn.ReLU())
        self.convgra = nn.Sequential(nn.Conv2d(inplanes1 * 2, inplanes1, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inplanes1),
                                     nn.ReLU())
        self.convsum = nn.Sequential(nn.Conv2d(inplanes1 * 4, inplanes1, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inplanes1),
                                     nn.ReLU())
        # self.conv53 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inplanes//2, inplanes*2, 1))
        self.conv7 = nn.Conv2d(16, 64, kernel_size=1, padding=0, stride=1, bias=False)
        #####################
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.fpt = F1.FPT(32)
        self.lastconv = nn.Sequential(convbn(320, 64, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))
        self.lastconv1 = nn.Sequential(convbn(384, 128, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 12, kernel_size=1, padding=0, stride=1, bias=False))
        self.graph = pvig_b_224_gelu()
        self.hilo = HiLo(inplanes1)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        ###########Graph
        out_gra = self.firstconv_gra(x)
        output_grap = self.graph(out_gra)
        # output_grap = self.convgra(output_grap)
        output_sum = torch.cat((output_skip, output_grap), dim=1)
        output_sum = self.convsum(output_sum)

        ############HiLo
        output_line = rearrange(output_sum, 'b c h w -> b (h w) c')
        out_hilo = self.hilo(output_line, 64, 64)
        # print(out_hilo.shape)
        ################DANet
        # PAM
        # feat1 = self.conv5a(output_sum)
        # sa_feat = self.sa(feat1)
        # sa_conv = self.conv51(sa_feat)
        # #先降维 CAM
        # feat2 = self.conv5c(output_sum)
        # sc_feat = self.raa(feat2)
        # sc_conv = self.conv52(sc_feat)
        # #sc_output = self.conv7(sc_conv)
        # feat_sum = sa_conv+sc_conv
        # output_line = rearrange(output_sum, 'b c h w -> b (h w) c')
        # feat_sum_hilo = self.hilo(output_line, 64, 64)
        # print("DA",feat_sum.size())
        ####################

        output_branch1 = self.branch1(out_hilo)
        # output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(out_hilo)
        # output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(out_hilo)
        # output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(out_hilo)
        # output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        # output_branch1, output_branch2, output_branch3, output_branch4 = self.fpt(output_branch1, output_branch2,
        #                                                                           output_branch3, output_branch4)
        # 金字塔上采样
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        gwc_feature = torch.cat((output_grap, out_hilo, output_raw, output_skip, output_branch4, output_branch3,
                                 output_branch2, output_branch1), 1)
        # print(output_branch1.shape)
        # feature1 = self.lastconv(output_branch1)
        # feature2 = self.lastconv(output_branch2)
        # feature3 = self.lastconv(output_branch3)
        # feature4 = self.lastconv(output_branch4)
        concat_feature = self.lastconv(gwc_feature)

        return concat_feature
