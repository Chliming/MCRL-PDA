import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Trans import Trans

heter_rwr = np.loadtxt(r'heter_rwr.txt')  # 4370 4370
heter_rwr = torch.from_numpy(np.array(heter_rwr,dtype='float32'))

d_graph_model = 4319  # 4370
nhead = 7  # (7,4319: 1, 7, 617)  (5,4370: 2, 5, 10, 19)
dim_feedforward = 2048
dropout = 0.5
num_layers = 3
model = Trans(d_graph_model, nhead, dim_feedforward, dropout, num_layers)

h = model(heter_rwr)
np.savetxt(r'trans_heter.txt', h.detach().numpy())
print(h)