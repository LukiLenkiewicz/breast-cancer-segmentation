from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self,
                input_channels: int = 3,
                kernel_size: int = 3,
                conv_stride: int = 1,
                padding: int = 1,
                dropout_rate: float = .25,
                layer_channels: list = [2, 4, 8, 16],
                mid_channels: int = 32,
                transpose_stride: int = 2,
                transpose_kernel_size: int = 3,
                ):
        super().__init__()

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.padding = padding
        self.dropout_rate= dropout_rate
        self.layer_channels = layer_channels
        self.mid_channels = mid_channels
        self.transpose_stride = transpose_stride
        self.transpose_kernel_size = transpose_kernel_size

        # contracting path
        self._generate_contracting_layers()
        self.max_pool = nn.MaxPool2d(2)

        # middle
        self.midlayer = ConvBlock(layer_channels[-1], mid_channels, self.kernel_size, self.conv_stride, self.padding)

        # expansive path
        self._generate_expansive_layers()

        # out layer
        self.out_layer = nn.Sequential(
            nn.Conv2d(layer_channels[0], 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        passes = deque()
        for layer in self.contracting_path:
            x = layer(x)
            passes.append(x.clone())
            x = self.max_pool(x)
            
        x = self.midlayer(x)

        for unconv, conv in zip(self.transpose, self.convs):
            copied = passes.pop()
            x = unconv(x)
            z = self.concat_embeddings(x, copied)
            z = conv(z)

        return self.out_layer(z)

    def _generate_contracting_layers(self):
        self.contracting_path = []
        prev_num_channels = self.input_channels
        for num_channels in self.layer_channels:

            # contracting layers      
            block = ConvBlock(prev_num_channels, num_channels, self.kernel_size, self.conv_stride, self.padding)
            dropout = nn.Dropout(self.dropout_rate)
            seq = nn.Sequential(block, dropout)

            self.contracting_path.append(seq)

            prev_num_channels = num_channels

    def _generate_expansive_layers(self):
        self.transpose = []
        self.convs = []
        prev_num_channels = self.mid_channels
        for num_channels in self.layer_channels[::-1]:            

            # transpose layers
            convt = nn.ConvTranspose2d(prev_num_channels, num_channels, self.transpose_kernel_size, self.conv_stride)
            self.transpose.append(convt)

            # conv layers
            dropout = nn.Dropout(self.dropout_rate)
            block = ConvBlock(prev_num_channels, num_channels, self.kernel_size, self.conv_stride, self.padding)
            seq = nn.Sequential(dropout, block)

            self.convs.append(seq)

            prev_num_channels = num_channels

    @staticmethod
    def concat_embeddings(x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
                                   nn.BatchNorm2d(out_channels), 
                                   nn.ReLU(),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
                                   nn.BatchNorm2d(out_channels), 
                                   nn.ReLU())
        
    def forward(self, x):
        x = self.block(x)
        return x
