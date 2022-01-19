"""
This file contains the pytorch model definitions for the dataset using 
the top 1000 select tags.
"""
import torch
from torch import nn
from torch.nn import Sequential, Linear, Dropout, ReLU, Sigmoid, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d, LeakyReLU
from torch import nn, einsum
from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 2, 1)#原先是2，3

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        # self.attn_1 = Attention(query_dim=51, context_dim=51, heads=2, dim_head=64)
        # self.fc = nn.Linear(51, 96, bias=False)
        self.audio_encoder = Sequential(
            #更改dropout层
            Conv2d(1, 128, kernel_size=4, stride=(2, 1), padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x48x48
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x24x24
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x12x12
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x6x6
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=(2, 1), padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x6x6
            Conv2d(128, 128, kernel_size=4, stride=1, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x3x3
            Dropout(.25),
            Flatten(),
        )
        self.fc_audio = Sequential(
            Linear(512, 256, bias= False),
            # Linear(1024, 512, bias=False),
            # Linear(512, 256, bias=False),
            # BatchNorm1d(256),
            # Sigmoid(),
            BatchNorm1d(256),
            ReLU(),  # 128x12x12
        )
        # self.fc_audio = Sequential(
        #     Linear(18432, 16384, bias=False),
        #     Linear(16384, 16384, bias=False),
        #     Linear(16384, 8192, bias=False),
        #     Linear(8192, 8192, bias=False),
        #     Linear(8192, 4096, bias=False),
        #     Linear(4096, 4096, bias=False),
        #     Linear(4096, 2048, bias=False),
        #     Linear(2048, 2048, bias=False),
        #     Linear(2048, 1024, bias=False),
        #     Linear(1024, 512, bias=False),
        #     Linear(512, 256, bias=False),
        #     BatchNorm1d(256),
        #     Sigmoid(),
        # )

    def forward(self, x):
        #加一个self attention
        # context = x.permute(0, 2, 1, 3, 4)
        # context = context.reshape(context.shape[0], 25, -1)
        # z = self.attn_1(context, context)
        # z = self.fc(z)
        # z = z.permute(0, 2, 1)
        # z = z.unsqueeze(1)

        z = self.audio_encoder(x)
        # z = self.audio_encoder(x)
        # z = self.fc_audio(z)
        # z_d = self.fc_audio(z)
        return z#, z_d


class AudioDecoder(nn.Module):
    def __init__(self):
        super(AudioDecoder, self).__init__()

        # self.fc_audio = Sequential(
        #     Linear(256, 512, bias=False),
        #     Linear(512, 1024, bias=False),
        #     Linear(1024, 2048, bias=False),
        #     Linear(2048, 2048, bias=False),
        #     Linear(2048, 4096, bias=False),
        #     Linear(4096, 4096, bias=False),
        #     Linear(4096, 8192, bias=False),
        #     Linear(8192, 8192, bias=False),
        #     Linear(8192, 16384, bias=False),
        #     Linear(16384, 16384, bias=False),
        #     Linear(16384, 18432, bias=False),
        # # 原先是1152,768,512,18432
        # # Dropout(0.25),
        #
        # )
        # self.unflatten = UnFlatten()
        #
        # self.convTrans_1 = ConvTranspose2d(128, 128, kernel_size=4, stride=(1, 1), padding=1, padding_mode='zeros')
        # self.convTrans_2 = ConvTranspose2d(128, 128, kernel_size=4, stride=(1, 2), padding=1, padding_mode='zeros')
        # self.convTrans_3 = ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        # self.convTrans_4 = ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        # self.convTrans_5 = ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        # self.convTrans_6 = ConvTranspose2d(128, 1, kernel_size=4, stride=(1, 2), padding=1, padding_mode='zeros')
        # self.attn_1 = Attention(query_dim = 96,context_dim= 51, heads=2, dim_head=64)
        self.audio_decoder = Sequential(
            # Linear(256, 512, bias=False),
            # Linear(512, 1024, bias=False),
            # Linear(256, 512, bias=False),  # 原先是1152,768,512
            # Dropout(0.25),
            UnFlatten(),
            # Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=1, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            ConvTranspose2d(128, 128, kernel_size=4, stride=(2, 1), padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            # Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            # Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            # Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            ConvTranspose2d(128, 1, kernel_size=4, stride=(2, 1) , padding=1, padding_mode='zeros'),
            BatchNorm2d(1),
            # Sigmoid(),
        )

    def forward(self, z, motion = None):
        # z = self.fc_audio(z)
        # z = self.unflatten(z)
        # print(z.shape)
        # z = self.convTrans_1(z)
        # print(z.shape)
        # z = self.convTrans_2(z)
        # print(z.shape)
        # z = self.convTrans_3(z)
        # print(z.shape)
        # z = self.convTrans_4(z)
        # print(z.shape)
        # z = self.convTrans_5(z)
        # print(z.shape)
        # z = self.convTrans_6(z)
        # print(z.shape)
        return self.audio_decoder(z)
        #添加类似残差神经网络的attention模块
        # z = self.audio_decoder(z)
        # context = motion.permute(0,2,1,3,4)
        # context = context.reshape(context.shape[0], 25, -1)
        # z = z.squeeze()
        # z = z.permute(0,2,1)
        # z = self.attn_1(z, context)
        # z = z.permute(0,2,1)
        # z = z.unsqueeze(1)
        # return z


class TagEncoder(nn.Module):
    def __init__(self):
        super(TagEncoder, self).__init__()

        self.tag_encoder = Sequential(
            Linear(1000, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 1152),
            BatchNorm1d(1152),
            ReLU(),
            Dropout(.25),
        )

        self.fc_tag = Sequential(
            Linear(1152, 1152, bias=False),
            Dropout(.25),
        )

    def forward(self, tags):
        z = self.tag_encoder(tags)
        z_d = self.fc_tag(z)
        return z, z_d


class TagDecoder(nn.Module):
    def __init__(self):
        super(TagDecoder, self).__init__()

        self.tag_decoder = Sequential(
            Linear(1152, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 1000),
            BatchNorm1d(1000),
            Sigmoid(),
        )

    def forward(self, z):
        return self.tag_decoder(z)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.encoder = Sequential( 
            Conv2d(1, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x48x48
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x24x24
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x12x12
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x6x6
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x3x3
            Dropout(.25),
            Flatten(),
        )

        self.fc = Sequential(
                Linear(1152, 1152),
                ReLU(),
                Linear(1152, 1000),
                Sigmoid(),
            )

    def forward(self, x):
        z = self.encoder(x)
        y = self.fc(z)
        return z, y
