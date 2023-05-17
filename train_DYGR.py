import torch
import random
import torch.optim as optim
from utils import *
import argparse
from model_DYGR import *
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.utils as U
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch.optim import lr_scheduler



parser = argparse.ArgumentParser()

# data
parser.add_argument('--suffix', type=str, default='MM_ER50_exp0', help='suffix for data')

parser.add_argument('--tr_num', type=int, default=50,
    help='No. of training trajectories, using all trajectories when None')

parser.add_argument('--va_num', type=int, default=10,
    help='No. of validation trajectories, using all trajectories when None')

parser.add_argument('--te_num', type=int, default=30,
    help='No. of test trajectories, using all trajectories when None')

parser.add_argument('--sample_freq', type=int, default=4,
    help='Sampling frequency of the trajectory')

parser.add_argument('--trajr_length', type=int, default=5,
    help='No. of time stamps in each trajectory, using all data when None')

parser.add_argument('--interlacing', type=bool, default=True,
    help='If the trajectories are interlacing when preparing dataset')

parser.add_argument('--Tstep', type=int, default=2, help='No. of steps for batched trajectories')

# model
parser.add_argument('--skip_first_edge_type', type=bool, default=False,
    help='If skip non-edges')

parser.add_argument('--gumbel_noise', type=bool, default=False, help='If includes gumbel noise')

parser.add_argument('--beta', type=float, default=0.5, help='Inverse temperature in softmax function')

parser.add_argument('--init_logits', type=str, default='random',
    help='initialization of logtis, (uniform, random)')

parser.add_argument('--hidden_channels', type=int, default=256)

parser.add_argument('--heads', type=int, default=1, help='number of filters')

#parser.add_argument('--prior', type=float, default=0.01)
parser.add_argument('--prior', type=float, default=0.001)
# training

parser.add_argument('--lr', type=float, default=0.0005, 
    help="Initial learning rate.")

parser.add_argument('--lr_z', type=float, default=0.1, 
    help="Learning rate for distribution estimation.")

parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--num_epoch', type=int, default=1000)

parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--filter', type=str, default='power',
         help='polynomial filter type (cheby, power)')

parser.add_argument('--K', type=int, default=4,
    help='trucation in the order for polynomial filters') 

parser.add_argument('--num-layers', type=int, default=1, help='number of GCN layers')

parser.add_argument('--skip', type=bool, default=True,
                     help='wether to use the skip connection, if None then it will be infered from data')

parser.add_argument('--skip_poly', type=bool, default=False)



parser.add_argument('--seed', type=int, default=5,
    help="random seed")

parser.add_argument('--device_id', type=int, default=0,
    help="device id")

args = parser.parse_args()
print(args)
seed=args.seed
if seed is None:
    seed=random.randint(100,10000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device('cuda:'+str(args.device_id) if torch.cuda.is_available() else 'cpu')
print('Device:', device)

if 'netsim' in args.suffix:
    x_tr, x_va, x_te, A = load_netsim_data(args)
else:
    x_tr, x_va, x_te, A = load_data(args) # loaded data has shape [batch, nodes, variables, time]

if args.K==0:
    args.K=1
    args.skip_poly=True


Tstep = args.Tstep
batch_size = args.batch_size
epochs = args.num_epoch
lr = args.lr
num_nodes = A.shape[0]
num_variables = x_tr.size(2)
num_edges = int(A.sum())
interlacing = args.interlacing
args.in_channels = num_variables
args.num_nodes = num_nodes
p=args.prior

train_loader = torch.utils.data.DataLoader(TrajrData(x_tr,Tstep,interlacing), batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(TrajrData(x_va,Tstep,interlacing), batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(TrajrData(x_te,Tstep,interlacing), batch_size=batch_size, shuffle=False)

if 'cheby' in args.filter:
    graph = ChebyGraphFilter(args).to(device)
elif 'power' in args.filter:
    graph = PowerGraphFilter(args).to(device)

dynamics = DynSurrogates(args).to(device)

optimizer = torch.optim.Adam(
        [{"params": graph.parameters(), "lr": args.lr_z}] +
        [{"params": dynamics.parameters(), "lr": args.lr}]
    )

criterion = torch.nn.MSELoss(reduction='mean')

def train(data_loader):
    graph.train()
    dynamics.train()
    loss_batch = 0
    num_datum = 0

    for x in data_loader:
        x = x.to(device)
        edge_prob, filter_bank = graph(edge_index)
        xpreds = dynamics(x[...,:-1], edge_index, edge_prob, filter_bank)
        loss = 0
        for i in range(len(xpreds)):
            loss += criterion(xpreds[i],x[...,1:])
            num_datum += xpreds[i].numel()
        loss_batch += loss.item() * x[...,1:].numel()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_batch = loss_batch/num_datum
    return loss_batch

def test(data_loader):
    graph.eval()
    dynamics.eval()

    loss_batch = 0
    num_datum = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            edge_prob, filter_bank = graph(edge_index)
            xpreds = dynamics(x[...,:-1], edge_index, edge_prob, filter_bank)
            loss = 0                                           
            for i in range(len(xpreds)):
                loss += criterion(xpreds[i],x[...,1:])
                num_datum += xpreds[i].numel()
            loss_batch += loss.item() * x[...,1:].numel()
    loss_batch = loss_batch/num_datum
    return loss_batch



if __name__ == '__main__':

    A_full = torch.ones(num_nodes,num_nodes)
    A_full = A_full - torch.diag(torch.diagonal(A_full))
    edge_index, _ = dense_to_sparse(A_full)
    edge_index = edge_index.to(device)

    best_val_loss = np.inf
    best_train_loss = np.inf
    best_mes_from_va = 0
    best_auc_from_va = 0
    best_acc_from_va = 0 
    loss_sum = 0 
    epochs = 0
    for epoch in range(1, args.num_epoch+1):
        loss_tr = train(train_loader)
        if(epoch%1 == 0):
            loss_va = test(valid_loader)
            loss_te = test(test_loader)

            A_soft, A_hard = generate_prediction(graph.logits.data,edge_index)
            auc, acc, pre = cal_accuracy(A, A_soft, A_hard, num_edges, epoch)            
        
            if loss_va < best_val_loss:
                best_train_loss =loss_tr
                best_val_loss = loss_va
                best_mes_from_va = loss_te
                best_auc_from_va = auc
                best_acc_from_va = acc

            print('Epoch: {:03d}'.format(epoch),
                'Train Loss: {:.8f}'.format(loss_tr),
                'Valid Loss: {:.8f}'.format(loss_va),
                'Picked AUC: {:.4f}'.format(best_auc_from_va),
                'Picked ACC: {:.4f}'.format(best_acc_from_va),
                'Current AUC: {:.4f}'.format(auc),
                'Current ACC: {:.4f}'.format(acc),
                'Current PRE: {:.4f}'.format(pre))
    import sys

    log_file = open('results_logs/dygr_'+args.suffix+'_'+str(args.sample_freq)+'_'+str(args.tr_num)+'_'+str(args.trajr_length)+'_'+str(args.K)+'_'+args.filter +'_'+ str(args.heads)+'_'+'{:1.0E}'.format(p)+'.txt', 'a')
    sys.stdout = log_file
    print('seed:{:08d}, auc:{:.4f}, acc:{:.4f}, last_acc:{:.4f}, train_loss:{:.8f}, test_loss:{:.8f}'.format(seed,best_auc_from_va,best_acc_from_va,auc,best_train_loss,best_mes_from_va))
    sys.stdout = sys.__stdout__
    log_file.close()
