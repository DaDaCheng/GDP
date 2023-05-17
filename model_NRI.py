'''
Re-implementation of NRI on the non-armotized setting.
The re-implementation are limited to the case of two edge types,

Implemented based on https://github.com/ethanfetaya/NRI 
Implemented based on https://github.com/loeweX/AmortizedCausalDiscovery
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_add

def sample_gumbel(shape, eps=1e-10):

    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))

def get_edge_prob(logits, gumbel_noise, beta=0.5, hard=False):
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


class MLPDecoder(nn.Module):
    def __init__(self, args):
        super(MLPDecoder, self).__init__()

        in_channels = args.in_channels
        hidden_channels = args.hidden_channels
        
        self.skip_first_edge_type = args.skip_first_edge_type
        self.dropout_prob = args.dropout
        self.preds = args.Tstep -1
        self.gumbel_noise = args.gumbel_noise

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * in_channels, hidden_channels) for _ in range(2)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(2)]
        )
        self.out_fc1 = nn.Linear(in_channels + hidden_channels, hidden_channels)
        self.out_fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.out_fc3 = nn.Linear(hidden_channels, in_channels)

    def mul(self, x, w):
        # (batch, edges, channels), (edges) -> (batch, edges, channels)
        return torch.einsum("bec,e->bec", x, w)

    def single_step_forward(self, x, edge_index, edge_prob):

        row, col = edge_index
        x0 = x.clone()

        # data has shape [batch, nodes, variables]
        # Node2edge
        pre_msg = torch.cat((x[:,row,:],x[:,col,:]),dim=-1)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx,2):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = self.mul(msg, edge_prob[i])
            all_msgs = msg if i==start_idx else all_msgs + msg

        agg_msgs = scatter_add(all_msgs, row, dim=1, dim_size=x0.size(1))

        # Skip Connection
        aug_inputs = torch.cat((x0,agg_msgs),dim=-1)

        # Output mlp
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)

        return x0 + pred

    def forward(self, inputs, edge_index, logits):
        last_pred = inputs[...,0]
        edge_prob = get_edge_prob(logits,self.gumbel_noise)
        preds = []
        for step in range(0,self.preds):
            last_pred = self.single_step_forward(last_pred, edge_index, edge_prob)
            preds.append(last_pred.unsqueeze(-1))

        preds = torch.cat(preds,dim=-1)
        return preds

