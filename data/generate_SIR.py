import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse
import networkx as nx
import argparse
import pickle


parser = argparse.ArgumentParser('Generate SIR simulation data')
parser.add_argument('--graph', type=str, default='BA')
parser.add_argument('--num-nodes', type=int, default=50,
                    help='Number of nodes in the simulation.')
parser.add_argument('--p', type=float, default=0.1, 
                    help='Connection/add connection probability In ER/NWS')
parser.add_argument('--k', type=int, default=2, 
                    help='Inital node degree in BA/NWS')

parser.add_argument('--exp_num', type=int, default=1, help='Number of repeated experiments')
parser.add_argument('--tr_num', type=int, default=100,
                    help='Number of train trajectories.')
parser.add_argument('--va_num', type=int, default=30,
                    help='Number of validation trajectories.')
parser.add_argument('--te_num', type=int, default=30,
                    help='Number of test trajectories.')


parser.add_argument('--infection_rate', type=float, default=0.6)
parser.add_argument('--recovery_rate', type=float, default=0.2)
parser.add_argument('--steps', type=int, default=300)


args = parser.parse_args()

beta = args.infection_rate
gamma = args.recovery_rate



# iteration
def simulation(steps):
    seed = np.random.randint(0, n)
    # x = np.zeros(n)
    # x[seed] = 1.0
    # s = np.ones(n)
    # s[seed] = 0.0
    # r = np.zeros(n)
    x = np.random.rand(n)
    s = 1 -x
    r = np.zeros(n)
    x_trajr = x.reshape(1,-1)
    s_trajr = s.reshape(1,-1)
    r_trajr = r.reshape(1,-1)
    for t in range(1,steps):
        infect = s*np.prod(1-beta*A*x,axis=1)
        recover = gamma * x
        x = x - recover + infect
        s = s - infect
        r = r + recover
        x_trajr = np.concatenate([x_trajr,x.reshape(1,-1)],axis=0)
        s_trajr = np.concatenate([s_trajr,s.reshape(1,-1)],axis=0)
        r_trajr = np.concatenate([r_trajr,r.reshape(1,-1)],axis=0)
    x_trajr = np.expand_dims(x_trajr, axis = -1)
    s_trajr = np.expand_dims(s_trajr, axis = -1)  
    r_trajr = np.expand_dims(r_trajr, axis = -1)
    x_trajr = np.concatenate((x_trajr,s_trajr,r_trajr),axis=-1)    
    return x_trajr


if __name__ == '__main__':
    assert args.graph in {'ER', 'NWS', 'BA'}, 'Unknown Graph Type'
    for exp_id in range(args.exp_num):
        n = args.num_nodes
        p = args.p
        k = args.k
        np.random.seed(exp_id)
        if args.graph in 'ER':
            G = nx.erdos_renyi_graph(n,p,seed=exp_id)
        elif args.graph in 'NWS':
            G = nx.newman_watts_strogatz_graph(n,k,p,seed=exp_id)
        elif args.graph in 'BA':
            G = nx.barabasi_albert_graph(n,k,seed=exp_id)

        A = nx.to_numpy_array(G)

        x_tr = np.zeros((args.tr_num,args.steps,n,3))
        for i in range(args.tr_num):
            print(f'Simulating training trajectory: {i+1:3d}/{args.tr_num:3d}')
            x_tr[i] = simulation(args.steps)
          
        x_va = np.zeros((args.va_num,args.steps,n,3))
        for i in range(args.va_num):
            print(f'Simulating  validation trajectory: {i+1:3d}/{args.va_num:3d}')
            x_va[i] = simulation(args.steps)

        x_te = np.zeros((args.te_num,args.steps,n,3))
        for i in range(args.te_num):
            print(f'Simulating test trajectory: {i+1:3d}/{args.te_num:3d}')
            x_te[i] = simulation(args.steps)

        A = nx.to_numpy_array(G)
        x_tr = x_tr.astype(np.float16)
        x_va = x_va.astype(np.float16)
        x_te = x_te.astype(np.float16)

        # save data, output data has shape [batch, time, nodes, variables]
        A = nx.to_numpy_array(G)
        result = [x_tr,x_va,x_te,A]
        data_path = 'SIR_' + args.graph + str(n) + '_exp' + str(exp_id) +'.pickle'
        with open(data_path, 'wb') as f:
            pickle.dump(result, f)

