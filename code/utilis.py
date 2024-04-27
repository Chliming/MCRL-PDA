import numpy as np
import torch


def feat_concat(embedding_hgc):
    feat = []  # feature matrix of total sample
    for i in range(embedding_hgc["circRNA"].shape[0]):
        for j in range(embedding_hgc["stress"].shape[0]):
            f = np.hstack((embedding_hgc["circRNA"][i], embedding_hgc["stress"][j]))
            feat.append(f)
    return feat


def min_max_normalization(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

