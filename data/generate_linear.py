import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse
import networkx as nx
import argparse
import pickle

import urllib.request
import io
import zipfile

parser = argparse.ArgumentParser('Generate SIR simulation data')
parser.add_argument('--graph', type=str, default='ER')
parser.add_argument('--num-nodes', type=int, default=30,
                    help='Number of balls in the simulation.')
parser.add_argument('--p', type=float, default=0.3, help='Connection probability in ER random graph')
parser.add_argument('--exp_num', type=int, default=1, help='Number of repeated experiments')
parser.add_argument('--tr_num', type=int, default=30,
                    help='Number of train trajectories.')
parser.add_argument('--va_num', type=int, default=10,
                    help='Number of validation trajectories.')
parser.add_argument('--te_num', type=int, default=10,
                    help='Number of test trajectories.')

parser.add_argument('--infection_rate', type=float, default=0.6)
parser.add_argument('--recovery_rate', type=float, default=0.5)
parser.add_argument('--steps', type=int, default=500)


args = parser.parse_args()

#assert args.graph == 'ER', 'unknown input graph type'

beta = args.infection_rate
gamma = args.recovery_rate






if __name__ == '__main__':
    for exp_id in range(args.exp_num):
        np.random.seed(exp_id)
        print(f'Simulating {exp_id+1:1d}/{args.exp_num:1d} experiment.')
        if args.graph == 'ER':
            G = nx.erdos_renyi_graph(n=args.num_nodes, p=args.p, seed=exp_id)

        elif args.graph == 'Grid':
            G = nx.grid_2d_graph(6,6)
        N = G.number_of_nodes()
        A = nx.to_numpy_array(G)
        #A = A + np.eye(N)
        D = np.sum(A,axis=1) + 1e-12
        D = 1/ np.sqrt(D)
        D = np.diag(D)
        A_norm = D@A@D
        
        # iteration
        def simulation(steps,A):
            X = np.random.rand(args.num_nodes)
            Xtrajr = X.reshape(1,-1)
            for t in range(1,steps):
                X = A@X
                Xtrajr = np.concatenate([Xtrajr,X.reshape(1,-1)],axis=0)
            Xtrajr = np.expand_dims(Xtrajr, axis = -1)
            return Xtrajr
        
        Xtr = np.zeros((args.tr_num,args.steps,args.num_nodes,1))
        for i in range(args.tr_num):
            print(f'Simulating training trajectory: {i+1:3d}/{args.tr_num:3d}')
            Xtr[i] = simulation(args.steps,A_norm.copy())
          
        Xva = np.zeros((args.va_num,args.steps,N,1))
        for i in range(args.va_num):
            print(f'Simulating  validation trajectory: {i+1:3d}/{args.va_num:3d}')
            Xva[i] = simulation(args.steps,A_norm.copy())

        Xte = np.zeros((args.te_num,args.steps,N,1))
        for i in range(args.te_num):
            print(f'Simulating test trajectory: {i+1:3d}/{args.te_num:3d}')
            Xte[i] = simulation(args.steps,A_norm.copy())

        A = nx.to_numpy_array(G)
        Xtr = Xtr.astype(np.float16)
        Xva = Xva.astype(np.float16)
        Xte = Xte.astype(np.float16)

        # save data, output data has shape [batch, time, nodes, variables]
        result = [Xtr,Xva,Xte,A]
        data_path = 'Lin_' + args.graph + str(args.num_nodes) + '_exp' + str(exp_id) +'.pickle'
        with open(data_path, 'wb') as f:
            pickle.dump(result, f)

