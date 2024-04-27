import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv, GINConv
import dgl.nn.pytorch as dglnn


class GCN(nn.Module):
    def __init__(self, C_dim, S_dim, hidden_dim):
        super(GCN, self).__init__()

        self.C_dim = C_dim
        self.S_dim = S_dim

        self.hidden_dim = hidden_dim

        self.HeteroConv1 = dglnn.HeteroGraphConv({
            'p_d': dglnn.GraphConv(self.C_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True),
            'd_p': dglnn.GraphConv(self.S_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True)
        },
            aggregate='sum')

        self.HeteroConv2 = dglnn.HeteroGraphConv({
            'p_d': dglnn.GraphConv(self.hidden_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True),
            'd_p': dglnn.GraphConv(self.hidden_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True)
        },
            aggregate='sum')

        self.HeteroConv3 = dglnn.HeteroGraphConv({
            'p_d': dglnn.GraphConv(self.hidden_dim, self.S_dim, activation=F.relu, allow_zero_in_degree=True),
            'd_p': dglnn.GraphConv(self.hidden_dim, self.C_dim, activation=F.relu, allow_zero_in_degree=True)
        },
            aggregate='sum')

    def forward(self, g, h):

        h1 = self.HeteroConv1(g, h)
        h2 = self.HeteroConv2(g, h1)
        h3 = self.HeteroConv3(g, h2)

        return h3


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.sigmoid(self.fc3(x))
        return x
