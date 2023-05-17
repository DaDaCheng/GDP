import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import to_dense_adj


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))

def get_laplacian(A):
    D = A.sum(dim=1) + 1e-6
    D = 1 / torch.sqrt(D)
    D = torch.diag(D)
    L = -D@A@D
    return L

def get_normalized_adjacency(A):
    D = A.sum(dim=1) + 1e-6
    D = 1 / torch.sqrt(D)
    D = torch.diag(D)
    A = D@A@D
    return A


def get_edge_prob(logits, gumbel_noise=False, beta=0.5, hard=False):
    if gumbel_noise:
        y = logits + sample_gumbel(logits.size()).to(logits.device)
    else:
        y = logits
    edge_prob_soft = torch.softmax(beta * y, dim=0)

    if hard:
        _, edge_prob_hard = torch.max(edge_prob_soft.data, dim=0)
        edge_prob_hard = F.one_hot(edge_prob_hard)
        edge_prob_hard = edge_prob_hard.permute(1,0)
        edge_prob = edge_prob_hard - edge_prob_soft.data + edge_prob_soft
    else:
        edge_prob = edge_prob_soft
    return edge_prob


class PowerGraphFilter(nn.Module):
    def __init__(self, args):
        super(PowerGraphFilter, self).__init__()

        K = args.K
        num_nodes = args.num_nodes
        heads = args.heads
        node_pairs = (num_nodes-1)*num_nodes

        self.K = K
        self.heads = heads
        if 'random' in args.init_logits:
            self.logits = nn.Parameter(torch.randn(torch.Size([2,node_pairs])))
        elif 'uniform' in args.init_logits:
            self.logits = nn.Parameter(torch.zeros(torch.Size([2,node_pairs])))
        
        self.theta0 = nn.ModuleList(
            [nn.Linear(K, 1) for _ in range(heads)]
        )
        self.theta1 = nn.ModuleList(
            [nn.Linear(K, 1) for _ in range(heads)]
        )
        self.gumbel_noise = args.gumbel_noise
        self.beta = args.beta

    def get_powers(self, A, edge_index):
        Ak = torch.eye(A.size(0)).to(A.device)
        Aks = []
        for k in range(1,self.K+1):
            Ak = A@Ak
        #for k in range(1,self.K+1):
            Aks.append(Ak[edge_index[0],edge_index[1]].view(-1,1))
        Aks = torch.cat(Aks,dim=1)
        return Aks

    def get_filters(self, A0, A1, edge_index):

        Aks0 = self.get_powers(A0, edge_index)
        Aks1 = self.get_powers(A1, edge_index)

        filter_bank = []
        for i in range(self.heads):
            filters = []
            filters.append(self.theta0[i](Aks0).squeeze(-1))
            filters.append(self.theta1[i](Aks1).squeeze(-1))
            filter_bank.append(filters)

        return filter_bank

    def forward(self, edge_index):

        edge_prob = get_edge_prob(self.logits, self.gumbel_noise, self.beta, False)
        A0 = to_dense_adj(edge_index,edge_attr=edge_prob[0]).squeeze(0)
        A1 = to_dense_adj(edge_index,edge_attr=edge_prob[1]).squeeze(0)

        A0 = get_normalized_adjacency(A0)
        A1 = get_normalized_adjacency(A1)

        filter_bank = self.get_filters(A0, A1, edge_index)

        return edge_prob, filter_bank


class ChebyGraphFilter(nn.Module):
    def __init__(self, args):
        super(ChebyGraphFilter, self).__init__()

        K = args.K
        num_nodes = args.num_nodes
        heads = args.heads
        node_pairs = (num_nodes-1)*num_nodes

        self.K = K
        self.heads = heads
        if 'random' in args.init_logits:
            self.logits = nn.Parameter(torch.randn(torch.Size([2,node_pairs])))
        elif 'uniform' in args.init_logits:
            self.logits = nn.Parameter(torch.zeros(torch.Size([2,node_pairs])))
        
        self.theta0 = nn.ModuleList(
            [nn.Linear(K, 1) for _ in range(heads)]
        )
        self.theta1 = nn.ModuleList(
            [nn.Linear(K, 1) for _ in range(heads)]
        )
        self.gumbel_noise = args.gumbel_noise
        self.beta = args.beta

    def get_powers(self, L, edge_index):    
        Tkm2 = torch.eye(L.size(0)).to(L.device)
        Tkm1 = L.clone()
        Tks = []
        Tks.append(L[edge_index[0],edge_index[1]].view(-1,1))
        for k in range(2,self.K+1):
            Tk = 2 * L@Tkm1 - Tkm2
            Tks.append(Tk[edge_index[0],edge_index[1]].view(-1,1))
            Tkm2 = Tkm1.clone()
            Tkm1 = Tk.clone()
        Tks = torch.cat(Tks,dim=1)
        return Tks

    def get_filters(self, L0, L1, edge_index):
        Lks0 = self.get_powers(L0, edge_index)
        Lks1 = self.get_powers(L1, edge_index)     
        filter_bank = []
        for i in range(self.heads):
            filters = []
            filters.append(self.theta0[i](Lks0).squeeze(-1))
            filters.append(self.theta1[i](Lks1).squeeze(-1))
            filter_bank.append(filters)

        return filter_bank

    def forward(self, edge_index):

        edge_prob = get_edge_prob(self.logits, self.gumbel_noise, self.beta, False)
        A0 = to_dense_adj(edge_index,edge_attr=edge_prob[0]).squeeze(0)
        A1 = to_dense_adj(edge_index,edge_attr=edge_prob[1]).squeeze(0)

        L0 = get_laplacian(A0)
        L1  = get_laplacian(A1)
        filter_bank = self.get_filters(L0, L1, edge_index)

        return edge_prob, filter_bank



class ConvLayer(nn.Module):
    def __init__(self, args):
        super(ConvLayer, self).__init__()

        in_channels = args.in_channels
        hidden_channels = args.hidden_channels
        self.skip = args.skip
        self.skip_first_edge_type = args.skip_first_edge_type
        self.dropout_prob = args.dropout
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*in_channels, hidden_channels) for _ in range(2)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(2)]
        )

        self.out_fc1 = nn.Linear(in_channels+hidden_channels, hidden_channels)
        self.out_fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.out_fc3 = nn.Linear(hidden_channels, in_channels)

        self.out_fc4 = nn.Linear(in_channels+hidden_channels, hidden_channels)
        self.out_fc5 = nn.Linear(hidden_channels, hidden_channels)
        self.out_fc6 = nn.Linear(hidden_channels, in_channels)

    # linear layer multiplication
    def mul(self, x, w):
        # (batch, edges, channels), (edges) -> (batch, edges, channels)
        return torch.einsum("bec,e->bec", x, w)

    def forward(self, x, edge_index, edge_prob):
  
        row, col = edge_index
        if self.skip:
            x0 = x.clone()
        # data has shape [batch, nodes, variables]
 
        pre_msg = torch.cat((x[:,row,:],x[:,col,:]),dim=-1)
        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx,2):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
            msg = F.relu(self.msg_fc2[i](msg))
            msg1 = self.mul(msg, edge_prob[i])
            all_msgs1 = msg1 if i==start_idx else all_msgs1 + msg1

        agg_msgs1 = scatter_add(all_msgs1, row, dim=1, dim_size=x0.size(1))
        aug_inputs1 = torch.cat((x0,agg_msgs1),dim=-1)

        #output mlp
        pred1 = F.dropout(F.relu(self.out_fc1(aug_inputs1)), p=self.dropout_prob, training=self.training)
        pred1 = F.dropout(F.relu(self.out_fc2(pred1)), p=self.dropout_prob, training=self.training)
        pred1 = self.out_fc3(pred1)


        if self.skip:
            pred1 += x0

        return pred1


class DynSurrogate(nn.Module):
    def __init__(self, args):
        super(DynSurrogate, self).__init__()

        in_channels = args.in_channels
        hidden_channels = args.hidden_channels
        out_channels = args.in_channels
        num_layers = args.num_layers

        self.skip = args.skip
        self.num_layers = num_layers
        self.pred_steps = args.Tstep - 1
        self.dropout_prob = args.dropout

        self.conv = torch.nn.ModuleList([])
        for i in range(num_layers):
            self.conv.append(ConvLayer(args))

    def single_step_forward(self, x, edge_index, filters):
        #[batch, nodes, channels]

        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index, filters)
            if i < self.num_layers-1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2,training=self.training)

        return x

    def forward(self, x, edge_index, filters):
        # input data has shape [batch, nodes, variables, time]
        last_pred = x[...,0]

        preds = []
        for step in range(0,self.pred_steps):

            last_pred = self.single_step_forward(last_pred, edge_index, filters)

            preds.append(last_pred.unsqueeze(-1))

        preds = torch.cat(preds,dim=-1)
        return preds


class DynSurrogates(nn.Module):
    def __init__(self, args):
        super(DynSurrogates, self).__init__()

        self.heads = args.heads
        self.model_A = DynSurrogate(args)    
        self.model_G = torch.nn.ModuleList([])
        for i in range(args.heads):
            if args.skip_poly:
                self.model_G.append(self.model_A)
            else:
                self.model_G.append(DynSurrogate(args))

    def forward(self, x, edge_index, edge_prob, filter_bank):
        # input data has shape [batch, nodes, variables, time]
        xpreds = []
        xpreds.append(self.model_A(x, edge_index, edge_prob))
        for i in range(self.heads):
            xpreds.append(self.model_G[i](x,edge_index,filter_bank[i]))
        
        return xpreds

