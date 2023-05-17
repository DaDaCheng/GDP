'''
Re-implementation of NRI on the non-armotized setting.
The re-implementation are limited to the case of two edge types,

Implemented based on https://github.com/ethanfetaya/NRI 
Implemented based on https://github.com/loeweX/AmortizedCausalDiscovery
'''

import torch
import torch.optim as optim
from utils import *
import argparse
from model_NRI import MLPDecoder, get_edge_prob
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import networkx as nx
import random

parser = argparse.ArgumentParser()
# data
parser.add_argument('--suffix', type=str, default='MM_ER50_exp0', help='suffix for data')

parser.add_argument('--tr_num', type=int, default=50,
    help='No. of training trajectories, using all trajectories when None')

parser.add_argument('--va_num', type=int, default=10,
    help='No. of validation trajectories, using all trajectories when None')

parser.add_argument('--te_num', type=int, default=30,
    help='No. of test trajectories, using all trajectories when None')

parser.add_argument('--sample_freq', type=int, default=1,
    help='Sampling frequency of the trajectory')

parser.add_argument('--trajr_length', type=int, default=10,
    help='No. of time stamps in each trajectory, using all data when None')

parser.add_argument('--interlacing', type=bool, default=True,
    help='If the trajectories are interlacing when preparing dataset')

parser.add_argument('--Tstep', type=int, default=2, help='No. of steps for batched trajectories')

# model
parser.add_argument('--skip_first_edge_type', type=bool, default=False,
    help='If skip non-edges')
parser.add_argument('--gumbel_noise', type=bool, default=True,help='If includes noise')
parser.add_argument('--init_logits', type=str, default='random',
    help='initialization of logtis, (uniform, random)')
parser.add_argument('--hidden_channels', type=int, default=256)

# training
parser.add_argument('--train', type=bool, default=True,
    help='If False, use test time adaption with a trained model')
parser.add_argument('--lr', type=float, default=0.0005, 
    help="Initial learning rate.")
parser.add_argument('--lr_z', type=float, default=0.1, 
    help="Learning rate for distribution estimation.")
parser.add_argument('--lr_logits', type=float, default=0.01,
    help="Learing rate for test time adaption")
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_tta_steps', type=int, default=100)
parser.add_argument('--lr_decay', type=int, default=200,help="After how epochs to decay LR by a factor of gamma.",)
parser.add_argument('--seed', type=int, default=0,help="random seed",)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor.")
args = parser.parse_args()
print(args.sample_freq)


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

print(x_tr.shape,x_va.shape,x_te.shape)

Tstep = args.Tstep
batch_size = args.batch_size
epochs = args.num_epoch
lr = args.lr
num_nodes = A.shape[0]
num_variables = x_tr.size(2)
num_edges = int(A.sum())
interlacing = args.interlacing
args.in_channels = num_variables

train_loader = torch.utils.data.DataLoader(TrajrData(x_tr,Tstep,interlacing), batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(TrajrData(x_va,Tstep,interlacing), batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(TrajrData(x_te,Tstep,interlacing), batch_size=batch_size, shuffle=False)



def train():

    def train_epoch(data_loader):
        model.train()
        loss_batch = 0
        num_datum = 0
        for x in data_loader:
            x = x.to(device)
            xpred = model(x[...,:-1],edge_index,logits)
            edge_prob = get_edge_prob(logits,gumbel_noise=False)
            kl_loss = kl_categorical_uniform(edge_prob, num_nodes)
            nll_loss = nll_gaussian(xpred,x[...,1:])
            
            A_soft = torch.softmax(0.5*logits,dim=0)
            D=torch.sum(torch.abs(A_soft[0]))*0.001
            
            
            
            loss = kl_loss + nll_loss+D
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_mse = criterion(x[...,1:], xpred)
            num_datum += xpred.numel()
            loss_batch += loss_mse.item()
        scheduler.step()
        loss_batch = loss_batch / num_datum
        return loss_batch

    def test_epoch(data_loader):
        model.eval()
        loss_batch = 0
        num_datum = 0
        with torch.no_grad():
            for x in data_loader:
                x = x.to(device)
                xpred = model(x[...,:-1],edge_index,logits)
                loss_mse = criterion(x[...,1:], xpred)
                loss_batch += loss_mse.item()
                num_datum += xpred.numel()
        loss_batch = loss_batch / num_datum
        return loss_batch

    optimizer = torch.optim.Adam(
            [{"params": logits, "lr": args.lr_z}] +
            [{"params": model.parameters(), "lr": args.lr}]
        )

    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay, gamma=args.gamma
    )
    data_path = './logs/nri' + args.suffix
    decoder_file = os.path.join(data_path+ "_decoder.pt")
    
    best_train_loss = np.inf
    best_val_loss = np.inf
    best_mes_from_va = 0
    best_auc_from_va = 0
    best_acc_from_va = 0 
    loss_sum = 0 
    epochs = 0
    patient=0
    
    for epoch in range(1, args.num_epoch+1):
        loss_tr = train_epoch(train_loader)
        if(epoch%1 == 0):
            loss_va = test_epoch(valid_loader)
            loss_te = test_epoch(test_loader)
            edge_prob = get_edge_prob(logits,gumbel_noise=False).clone().detach()
            A_soft, A_hard = generate_prediction_nri(edge_prob,edge_index)
            auc, acc, pre = cal_accuracy(A, A_soft, A_hard, num_edges)
            if loss_va < best_val_loss:
                best_train_loss= loss_tr
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

    return best_auc_from_va,best_acc_from_va,auc,best_train_loss,best_mes_from_va


def test_time_adaption():
    def tta_epoch(data_loader):
        model.eval()
        loss_batch = 0
        num_datum = 0
        for x in data_loader:
            for i in range(args.num_tta_steps):
                x = x.to(device)
                xpred = model(x[...,:-1],edge_index,logits)
                edge_prob = get_edge_prob(logits,gumbel_noise=False)
                kl_loss = kl_categorical_uniform(edge_prob, num_nodes)
                nll_loss = nll_gaussian(xpred,x[...,1:])
                loss = kl_loss + nll_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_mse = criterion(x[...,1:], xpred)
                num_datum += xpred.numel()
                loss_batch += loss_mse.item()
        loss_batch = loss_batch / num_datum
        return loss_batch

    data_path = './logs/nri' + args.suffix
    decoder_file = os.path.join(data_path + "_decoder.pt")
    model.load_state_dict(torch.load(decoder_file))
    optimizer = torch.optim.Adam(
            [{"params": logits, "lr": args.lr_logits}]
        )
    criterion = torch.nn.MSELoss(reduction='sum')    

    for epoch in range(1, args.num_tta_steps + 1):
        loss_te = tta_epoch(train_loader)
        if(epoch%1 == 0):
            edge_prob = get_edge_prob(logits,gumbel_noise=False).clone().detach()
            A_soft, A_hard = generate_prediction_nri(edge_prob,edge_index)
            auc, acc = cal_accuracy(A, A_soft, A_hard)

            print('Epoch: {:03d}'.format(epoch),
                'Test Loss: {:.8f}'.format(loss_te),
                'Current AUC: {:.4f}'.format(auc),
                'Current ACC: {:.4f}'.format(acc))


if __name__ == '__main__':

    A_full = torch.ones(num_nodes,num_nodes)
    A_full = A_full - torch.diag(torch.diagonal(A_full))
    edge_index, _ = dense_to_sparse(A_full)

    edge_index = edge_index.to(device)
    node_pairs = num_nodes*(num_nodes-1)

    if 'uniform' in args.init_logits:
        logits = torch.zeros(torch.Size([2,node_pairs]),requires_grad=True,device=device)
    elif 'random' in args.init_logits:
        logits = torch.randn(torch.Size([2,node_pairs]),requires_grad=True,device=device)

    model = MLPDecoder(args).to(device)
    criterion = torch.nn.MSELoss(reduction='sum')

    if args.train:
        best_auc_from_va,best_acc_from_va,last_auc,best_train_loss,best_mes_from_va=train()
    else:
        test_time_adaption()
        
        
    import sys

    # Open a file for logging
    log_file = open('results_logs/nri_'+args.suffix+'_'+str(args.sample_freq)+'_'+str(args.tr_num)+'_'+str(args.trajr_length)+'.txt', 'a')
    sys.stdout = log_file
    print('seed:{:08d}, auc:{:.4f}, acc:{:.4f}, last_acc:{:.4f}, train_loss:{:.8f}, test_loss:{:.8f}'.format(seed,best_auc_from_va,best_acc_from_va,last_auc,best_train_loss,best_mes_from_va))
    sys.stdout = sys.__stdout__
    log_file.close()

            
            
