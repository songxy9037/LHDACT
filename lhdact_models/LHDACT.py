import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange
from models import Transformer
import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from .submodule import LHDACTbone
from Transformer_ import Transformer_

class LHDACT(nn.Module):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, output_nc=2):
        super(LHDACT, self).__init__()
        self.Transformer = Transformer(token_len=4, decdepth=4, in_channel=64)
        self.feature_ex = LHDACTbone()
        # self.fea_nogra = feature_extraction_nogra()
        # self.fea_nocat = feature_extraction_nocat()
        # self.fea_nomsp = feature_extraction_nomsp()
        # self.fea_nofpt = feature_extraction_nofpt()
        # self.fea_nohilo = feature_extraction_nohilo()

        self.conv_final = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32,32,3,1,1),
                                    nn.BatchNorm2d(32))
        self.sigmoid = nn.Sigmoid()
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.classifier = TwoLayerConv2d(in_channels=64, out_channels=output_nc)


    def forward(self, x1, x2):
        outputs = []
        # forward backbone resnet
        feature_A = self.feature_ex(x1)
        feature_B = self.feature_ex(x2)

        #  forward tokenzier
        feature_A_de, feature_B_de = self.Transformer(feature_A, feature_B)

        # feature differencing
        x_abs = torch.abs(feature_A_de - feature_B_de)

        x = self.upsamplex4(x_abs)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        outputs.append(x)
        return outputs

