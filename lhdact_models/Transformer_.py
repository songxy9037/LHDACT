import torch
import torch.nn as nn

from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import functools
from einops import rearrange
import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d


class Transformer_(nn.Module):
    def __init__(self, token_len=4, enc_depth=1, dec_depth=8, in_channel=64,
                 dim_head=64, decoder_dim_head=64, decoder_softmax=True):
        super(Transformer, self).__init__()
        self.token_len=token_len
        dim = in_channel
        mlp_dim = 2*dim
        self.conv_a = nn.Conv2d(in_channels=in_channel, out_channels=token_len, kernel_size=1,
                                padding=0, bias=False)
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len, in_channel))

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def transformer_encoder(self, x):
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, x1, x2):
        token_1 = self._forward_semantic_tokens(x1)
        token_2 = self._forward_semantic_tokens(x2)
        token = torch.cat([token_1, token_2], dim=1)
        token_new = self._forward_transformer(token)
        token_n1, token_n2 = token_new.chunk(2, dim=1)
        x1 = self._forward_transformer_decoder(x1, token_n1)
        x2 = self_forward_transformer_decoder(x2, token_n2)
        return x1, x2