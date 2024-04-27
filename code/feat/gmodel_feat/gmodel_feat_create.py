from Graph_create import get_ass,Graph_create
import torch
import torch.nn as nn
import numpy as np
from model import GCN
import pandas as pd

file_path = r'ass.txt'

piRNA_disease = get_ass(file_path)

piRNA_feat = np.loadtxt(r'PS.txt')
dis_feat = np.loadtxt(r'DS.txt')

piRNA_feat = np.array(piRNA_feat, dtype='float32')
dis_feat = np.array(dis_feat, dtype='float32')

piRNA_feat = torch.from_numpy(piRNA_feat)
dis_feat = torch.from_numpy(dis_feat)

graph, graph_h = Graph_create(piRNA_disease, piRNA_feat, dis_feat)

model = GCN(C_dim=4349, S_dim=21, hidden_dim=64)   # 4349,21

h = model(graph, graph_h)

np.save(r'gmodel_feat_piR.npy', h['piRNA'].detach().numpy())
np.save(r'gmodel_feat_dis.npy', h['disease'].detach().numpy())