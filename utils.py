import pickle
import torch
import os.path
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import matplotlib.pyplot as plt

def load_data(args):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = './data/' + args.suffix +'.pickle'
    data_path = os.path.join(cur_dir, data_path)

    with open(data_path, 'rb') as f:
        x_tr, x_va, x_te, A = pickle.load(f)
    x_tr = torch.from_numpy(x_tr).to(torch.float32)
    x_va = torch.from_numpy(x_va).to(torch.float32)
    x_te = torch.from_numpy(x_te).to(torch.float32)
    A = (np.rint(A)).astype(int)


    # number of trajectories in train/validation/test set
    if args.tr_num is not None:
        assert x_tr.size(0) >= args.tr_num, 'No sufficent train data!'
        x_tr = x_tr[:args.tr_num,...]
    if args.va_num is not None:
        assert x_va.size(0) >= args.va_num, 'No sufficent validation data!'
        x_va = x_va[:args.va_num,...]
    if args.te_num is not None:
        assert x_te.size(0) >= args.te_num, 'No sufficent test data!'
        x_te = x_te[:args.te_num,...]


    # sample subsequence for larger obvervation interval
    if args.sample_freq != 1:
        x_tr = x_tr[:,0::args.sample_freq,:,:]
        x_va = x_va[:,0::args.sample_freq,:,:]
        x_te = x_te[:,0::args.sample_freq,:,:]



    # trajectory length
    if args.trajr_length is not None:
        assert x_tr.size(1) >= args.trajr_length, 'Not enough trajectory length!' 
        x_tr = x_tr[:,:args.trajr_length,:,:]
        x_va = x_va[:,:args.trajr_length,:,:]
        x_te = x_te[:,:args.trajr_length,:,:]

    #Normalize each system state dimension to [-1, 1]
    for i in range(x_tr.size(-1)):
        xmax = x_tr[:,:,:,i].max()
        xmin = x_tr[:,:,:,i].min()

        x_tr[:,:,:,i] = (x_tr[:,:,:,i] - xmin) * 2 / (xmax - xmin) -1
        x_va[:,:,:,i] = (x_va[:,:,:,i] - xmin) * 2 / (xmax - xmin) -1
        x_te[:,:,:,i] = (x_te[:,:,:,i] - xmin) * 2 / (xmax - xmin) -1
    
    # input data has shape [batch, time, nodes, variables]
    x_tr = x_tr.permute(0,2,3,1)
    x_va = x_va.permute(0,2,3,1)
    x_te = x_te.permute(0,2,3,1)
    # data has shape [batch, nodes, variables, time]

    print('Training Trajectories: {:03d}'.format(x_tr.size(0)),
        'Trajectory length: {:03d}'.format(x_tr.size(3)))

    return x_tr, x_va, x_te, A



def load_netsim_data(args,batch_size=1, datadir="data"):
    print("Loading data from {}".format(datadir))

    subject_id = [1, 2, 3, 4, 5]

    print("Loading data for subjects ", subject_id)

    loc_train = torch.zeros(len(subject_id), 15, 200)
    edges_train = torch.zeros(len(subject_id), 15, 15)

    for idx, elem in enumerate(subject_id):
        fileName = "sim3_subject_%s.npz" % (elem)
        ld = np.load(os.path.join(datadir, "netsim", fileName))
        loc_train[idx] = torch.FloatTensor(ld["X_np"])
        edges_train[idx] = torch.LongTensor(ld["Gref"])

    # [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = loc_train.unsqueeze(-1)

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    loc_train = normalize(loc_train, loc_min, loc_max)

    if args.sample_freq != 1:
        loc_train = loc_train[:,:,0::args.sample_freq]
    

    loc_train = loc_train.permute(0,1,3,2)
    

    x_tr = loc_train.clone()
    x_va = loc_train.clone()
    x_te = loc_train.clone()

    # Exclude self edges
    A = edges_train[0].int().numpy()
    A = A - np.diag(np.diagonal(A))
    print('Training Trajectories: {:03d}'.format(x_tr.size(0)),
        'Trajectory length: {:03d}'.format(x_tr.size(3)))
    return x_tr, x_va, x_te, A

def normalize(x, x_min, x_max):
    return (x - x_min) * 2 / (x_max - x_min) - 1


def cal_accuracy(A, A_soft, A_hard, num_edges, epoch=0):
    mask = ~np.eye(A.shape[0],dtype=bool)
    off_diag_idx = np.where(mask)

    scores = A_soft[off_diag_idx]
    labels = A[off_diag_idx]

    #auc
    auc = roc_auc_score(labels, scores)
    if auc < 0.5:
        auc = 1- auc 
        scores = - scores
        A_hard = 1-A_hard
    #acc
    acc = (labels == A_hard[off_diag_idx]).mean()

    ind = np.argsort(scores)
    pre = labels[ind[-num_edges:]].mean()

    return auc, acc, pre


class TrajrData(Dataset):
    def __init__(self, data, Tstep, interlacing=True):
        self.data = data
        # data has shape [batch, nodes, variables, time]
        self.interlacing = interlacing
        if interlacing:
            self.Tout = self.data.shape[-1] - Tstep +1 #steps for reccurent output
        else:
            assert self.data.shape[-1]%Tstep == 0, 'Trajectory length must be integer multiple of Tstep'
            self.Tout = int(np.ceil(self.data.shape[-1]/Tstep))
        self.Tstep = Tstep
        self.batch = self.data.shape[0]
        self.datalen = self.batch*self.Tout
    def __len__(self):
        return self.datalen
    def __getitem__(self, idx):
        i, j = idx//self.Tout, idx%self.Tout #i: batch, j: start time step
        if self.interlacing:
            sample = self.data[i,:,:,j:j+self.Tstep]
        else:
            start_ind = j*self.Tstep
            sample = self.data[i,:,:,start_ind:start_ind+self.Tstep]
        return sample



def kl_categorical_uniform(
    preds, num_atoms, add_const=False, eps=1e-16
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    kl_div = preds * (torch.log(preds + eps))
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / num_atoms

def nll_gaussian(preds, target, variance=5e-7, add_const=False):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const

    return neg_log_p.sum() / (target.size(0) * target.size(1))


def generate_prediction(edge_prob, edge_index):
    A_hard = (edge_prob[1] > edge_prob[0]).long()
    A_soft = torch.softmax(0.5*edge_prob,dim=0)
    A_soft = to_dense_adj(edge_index, edge_attr = A_soft[1]).cpu().squeeze(0).numpy()
    A_hard = to_dense_adj(edge_index, edge_attr = A_hard).cpu().squeeze(0).numpy()
    return A_soft, A_hard

def generate_prediction_nri(edge_prob, edge_index):
    A_hard = (edge_prob[1] > edge_prob[0]).long()
    A_soft = torch.softmax(0.5*edge_prob,dim=0)
    A_soft = to_dense_adj(edge_index, edge_attr = A_soft[1]).cpu().squeeze(0).numpy()
    A_hard = to_dense_adj(edge_index, edge_attr = A_hard).cpu().squeeze(0).numpy()
    return A_soft, A_hard
   
