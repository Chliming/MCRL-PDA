from DilatedCNN import DilatedCNN
import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd

stime = time.time()
in_channels = 1
out_channels = 8
dcn_kernel_size = (1,2)
pool_kernel_size = (1,2)
dilation = [1,2,3]
dropout = [0.5]

view1 = DilatedCNN(in_channels, out_channels, dcn_kernel_size, pool_kernel_size, 2, dropout[0])

x1 = pd.read_csv(r'dcnn_feat_ld_label_bal.csv',index_col=0)
x1 = x1.iloc[:,:-1]
x1 = np.array(x1)
x1 = torch.from_numpy(x1)
x1 = x1.unsqueeze(0)
x1 = x1.to(torch.float)


# 多视图特征更新
view1_feat = view1(x1)

np.save(r'view2_feat.npy', view1_feat.detach().numpy())
etime=time.time()
t = etime-stime
print('已完成feat2，用时{:.2f}'.format(t))