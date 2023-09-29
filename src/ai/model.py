from typing import Iterable
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        layer_channels: list = [16, 32, 64, 128],
        mid_channel_size: int = 256
    ):
        super().__init__()

        contracting_layer_channels = [input_channels] + layer_channels
        self.contracting_layer = self._generate_contracting_layer(contracting_layer_channels)
        
        self.mid_channel = ResidualBlock(layer_channels[-1], mid_channel_size)

        expansive_layer_channels = layer_channels + [mid_channel_size]
        expansive_layer_channels = expansive_layer_channels[::-1]
        self.expansive_layer = self._generate_expansive_layer(expansive_layer_channels)

        self.out_layer = nn.Conv2d(layer_channels[0], output_channels, kernel_size=1)

    def forward(self, x):
        skipped_tensors = []

        for layer in self.contracting_layer:
            x, skipped = layer(x)
            skipped_tensors.append(skipped)

        x = self.mid_channel(x)

        for layer, skipped in zip(self.expansive_layer, skipped_tensors[::-1]):
            x = layer(x, skipped)

        return self.out_layer(x)


    def _generate_contracting_layer(self, contracting_layer_channels: Iterable) -> nn.Sequential:
        encoder_blocks = []

        for i in range(len(contracting_layer_channels)-1):
            in_channels = contracting_layer_channels[i]
            out_channels = contracting_layer_channels[i+1]
            block = EncoderBlock(in_channels, out_channels)
            encoder_blocks.append(block)

        contracting_layer = nn.Sequential(*encoder_blocks)
        return contracting_layer
    

    def _generate_expansive_layer(self, expansive_layer_channels: Iterable) -> nn.Sequential:
        decoder_blocks = []

        for i in range(len(expansive_layer_channels)-1):
            in_channels = expansive_layer_channels[i]
            out_channels = expansive_layer_channels[i+1]
            block = DecoderBlock(in_channels, out_channels)
            decoder_blocks.append(block)

        expansive_layer = nn.Sequential(*decoder_blocks)
        return expansive_layer


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate=0.2) -> tuple[torch.Tensor, torch.Tensor]:
        super().__init__()

        self.block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        skipped = self.block(x)
        pooled = self.pool(skipped)
        pooled = self.dropout(pooled)
        return pooled, skipped 


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate=0.2) -> torch.Tensor:
        super().__init__()

        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = ResidualBlock(out_channels*2, out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, skipped):
        x = self.up_sample(x)
        concat = torch.cat([x, skipped], dim=1)
        concat = self.dropout(concat)
        y = self.block(concat)
        return y


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1, padding: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 5, padding: int = 2):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=2)

        self.block = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(output_channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        residual = x
        x = self.block(x)
        x = x + residual
        x = self.relu(x)

        return x
