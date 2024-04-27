from DilatedCNN import DilatedCNN
import numpy as np
import torch
import torch.nn as nn
import time

stime = time.time()
in_channels = 8
out_channels = 1  # 不改
dcn_kernel_size = (1,2)
pool_kernel_size = (1,2)
dilation = [1,2,3]
dropout = [0.5]

view1 = DilatedCNN(in_channels, 1, dcn_kernel_size, pool_kernel_size, 1, dropout[0])  # d=1 out_channels=1

x1 = np.load(r'view4_feat.npy')
x1 = torch.from_numpy(x1)
x1 = x1.to(torch.float)


# 多视图特征更新
view1_feat = view1(x1)

np.save(r'view5_feat.npy', view1_feat.detach().numpy())
etime=time.time()
t = etime-stime
print('已完成feat5，用时{:.2f}'.format(t))