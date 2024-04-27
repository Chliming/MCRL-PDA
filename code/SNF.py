import time

import numpy as np
import math
import random
import pandas as pd


def new_normalization(w):
    m = w.shape[0]
    p = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1 / 2
            elif np.sum(w[i, :]) - w[i, i] != 0:
                p[i][j] = w[i, j] / (2 * (np.sum(w[i, :]) - w[i, i]))
    return p


def KNN_kernel(S, k):
    n = S.shape[0]
    S_knn = np.zeros([n, n])
    for i in range(n):
        sort_index = np.argsort(S[i, :])  # 从小到大的索引
        for j in sort_index[n - k:n]:
            if np.sum(S[i, sort_index[n - k:n]]) != 0:
                S_knn[i][j] = S[i][j] / (np.sum(S[i, sort_index[n - k:n]]))
    return S_knn


def MiRNA_updating(S1, S2, S4, P1, P2, P4):
    it = 0
    P = (P1 + P2 + P4) / 3
    dif = 1
    while dif > 0.0000001:
        it = it + 1
        P111 = np.dot(np.dot(S1, (P2 + P4) / 2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, (P1 + P4) / 2), S2.T)
        P222 = new_normalization(P222)
        P444 = np.dot(np.dot(S4, (P1 + P2) / 2), S4.T)
        P444 = new_normalization(P444)
        P1 = P111
        P2 = P222
        P4 = P444
        P_New = (P1 + P2 + P4) / 3
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P


def disease_updating(S1, S2, P1, P2):
    it = 0
    P = (P1 + P2) / 2
    dif = 1
    while dif > 0.0000001:
        it = it + 1
        P111 = np.dot(np.dot(S1, P2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, P1), S2.T)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1 + P2) / 2
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb2", it)
    return P


print('开始......')
stime = time.time()
A = pd.read_csv(r"adj_index.csv", index_col=0)

k1, k2, k3, k4, k5 = 1546, 1, 1546, 9, 1  # 每个相似性网络的所有行的大于0.9的个数均值 SNF的K值获取

piRNA_sim1 = np.loadtxt(r"DFunc.txt")
piRNA_sim2 = np.loadtxt(r"seq_sim.txt")
GIP_p_sim = np.loadtxt(r"DGIPKp.txt")

GIP_d_sim = np.loadtxt(r"GIPKd.txt")
disease_sim1 = np.load(r"sem.npy")

print('融合rna')
m1 = new_normalization(piRNA_sim1)
m2 = new_normalization(piRNA_sim2)
m4 = new_normalization(GIP_p_sim)

print('融合rna：knn1')
Sm_1 = KNN_kernel(piRNA_sim1, k1)
print('融合rna：knn2')
Sm_2 = KNN_kernel(piRNA_sim2, k2)
print('融合rna：knn3')
Sm_4 = KNN_kernel(GIP_p_sim, k3)
print('融合rna：特征更新')
Pm = MiRNA_updating(Sm_1, Sm_2, Sm_4, m1, m2, m4)
Pm_final = (Pm + Pm.T) / 2

print('融合disease')
d1 = new_normalization(disease_sim1)
d2 = new_normalization(GIP_d_sim)

print('融合disease：knn4')
Sd_1 = KNN_kernel(disease_sim1, k4)
print('融合disease：knn5')
Sd_2 = KNN_kernel(GIP_d_sim, k5)
print('融合disease：特征更新')
Pd = disease_updating(Sd_1, Sd_2, d1, d2)
Pd_final = (Pd + Pd.T) / 2

np.savetxt(r"PS.txt", Pm_final)
np.savetxt(r"DS.txt", Pd_final)
etime = time.time()
t = etime - stime
print('已保存...用时{:.2f}'.format(t))
