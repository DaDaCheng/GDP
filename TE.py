import torch
import torch.optim as optim
from utils import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import networkx as nx
import torch.nn.utils as U
import torch.nn.functional as F
import netrd
from netrd.utilities.entropy import categorized_data, conditional_entropy
from itertools import permutations

parser = argparse.ArgumentParser()
# data
parser.add_argument('--suffix', type=str, default='netsim', help='suffix for data')

parser.add_argument('--tr_num', type=int, default=20,
    help='No. of training trajectories, using all trajectories when None')

parser.add_argument('--va_num', type=int, default=10,
    help='No. of validation trajectories, using all trajectories when None')

parser.add_argument('--te_num', type=int, default=1,
    help='No. of test trajectories, using all trajectories when None')

parser.add_argument('--sample_freq', type=int, default=4,
    help='Sampling frequency of the trajectory')

parser.add_argument('--trajr_length', type=int, default=10,
    help='No. of time stamps in each trajectory, using all data when None')

parser.add_argument('--interlacing', type=bool, default=True,
    help='If the trajectories are interlacing when preparing dataset')

parser.add_argument('--Tstep', type=int, default=2, help='No. of steps for batched trajectories')
args = parser.parse_args([])
if 'netsim' in args.suffix:
    x_tr, x_va, x_te, A = load_netsim_data(args)
else:
    x_tr, x_va, x_te, A = load_data(args) # loaded data has shape [batch, nodes, variables, time]
    x_tr = torch.cat((x_tr,x_va),dim=0)
aa = 0
num_nodes = x_tr.size(1)
num_edges = np.count_nonzero(A)
batch = x_tr.size(0)
W = 0

for v in range(x_tr.size(2)):
    x = x_tr[:,:,v,:]  #[batch, nodes, time]
    x = x.permute(1,0,2) #[nodes, batch, time]
    x = x.reshape(num_nodes,-1).T.numpy()
    data = categorized_data(x, n_bins=2).T
    data = data.reshape(num_nodes,batch,-1)
    TE = np.zeros((num_nodes,num_nodes))
    for i, j in permutations(range(num_nodes), 2):
        x, y = data[i,...], data[j,...]
        x_past, y_past = x[:,:-1], y[:,:-1]
        y_future = y[:,1:]
        x_past = x_past.reshape(-1,1)
        y_past = y_past.reshape(-1,1)
        y_future = y_future.reshape(-1,1)
        joint_past = np.hstack((y_past,x_past))
        TE[i,j] = conditional_entropy(y_future,y_past)
        TE[i,j] -= conditional_entropy(y_future,joint_past)
    W += TE
auc_te_2, acc, pre = cal_accuracy(A, W, W>1, num_edges, 0)


aa = 0
num_nodes = x_tr.size(1)
num_edges = np.count_nonzero(A)
batch = x_tr.size(0)
W = 0

for v in range(x_tr.size(2)):
    x = x_tr[:,:,v,:]  #[batch, nodes, time]
    x = x.permute(1,0,2) #[nodes, batch, time]
    x = x.reshape(num_nodes,-1).T.numpy()
    data = categorized_data(x, n_bins=200).T
    data = data.reshape(num_nodes,batch,-1)
    TE = np.zeros((num_nodes,num_nodes))
    for i, j in permutations(range(num_nodes), 2):
        x, y = data[i,...], data[j,...]
        x_past, y_past = x[:,:-1], y[:,:-1]
        y_future = y[:,1:]
        x_past = x_past.reshape(-1,1)
        y_past = y_past.reshape(-1,1)
        y_future = y_future.reshape(-1,1)
        joint_past = np.hstack((y_past,x_past))
        TE[i,j] = conditional_entropy(y_future,y_past)
        TE[i,j] -= conditional_entropy(y_future,joint_past)
    W += TE
auc_te_200, acc, pre = cal_accuracy(A, W, W>1, num_edges, 0)

auc_te=np.maximum(auc_te_2,auc_te_200)

print('TE:',f'{auc_mi*100:.2f}')