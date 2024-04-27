import numpy as np
import torch
import time

stime = time.time()
view1_feat = np.load(r'view1_feat.npy')
view2_feat = np.load(r'view2_feat.npy')
view3_feat = np.load(r'view3_feat.npy')

view4_feat = np.concatenate((view1_feat, view2_feat, view3_feat),axis=2)  # 16 19 27

view4_feat = torch.from_numpy(view4_feat)

np.save(r'view4_feat.npy', view4_feat)
etime=time.time()
t = etime-stime
print('已完成feat4，用时{:.2f}'.format(t))