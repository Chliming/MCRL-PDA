import numpy as np
import torch
import torch.nn as nn


class DilatedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dcn_kernel_size, pool_kernel_size, dilation,
                 dropout):
        super(DilatedCNN, self).__init__()

        # 卷积池化层，提取多视图特征
        self.conv = nn.Conv2d(in_channels, out_channels, dcn_kernel_size, stride=1, padding='same', dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # 多视图特征提取
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        return x
