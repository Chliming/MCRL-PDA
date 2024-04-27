from DilatedCNN import DilatedCNN
import numpy as np
import torch
import torch.nn as nn
import time

stime = time.time()
view5_feat = np.load(r'view5_feat.npy')

fc = nn.Linear(in_features=view5_feat.shape[2], out_features=4319)  # or 4370 for piRDieasev1.0
view5_feat = torch.from_numpy(view5_feat)
view5_feat = view5_feat.squeeze() # 19 13

view6_feat = fc(view5_feat)
relu = nn.ReLU(inplace=True)
embedding_dcn = relu(view6_feat)
print(embedding_dcn)

np.save(r'dcnn_feat.npy',embedding_dcn.cpu().detach().numpy())
etime=time.time()
t = etime-stime
print('已完成dcnn_feat，用时{:.2f}'.format(t))