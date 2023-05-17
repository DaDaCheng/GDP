"""Based on https://github.com/ethanfetaya/NRI """
"""Based on https://github.com/loeweX/AmortizedCausalDiscovery"""
"""Based on https://github.com/tailintalent/causal"""

import numpy as np
import os
import time
import argparse
import networkx as nx
import pickle
import kuramoto
import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default='BA')
    parser.add_argument('--num-nodes', type=int, default=50,
                        help='Number of nodes in the simulation.')
    parser.add_argument('--p', type=float, default=0.1, 
                        help='Connection/add connection probability In ER/NWS')
    parser.add_argument('--k', type=int, default=2, 
                        help='Inital node degree in BA/NWS')

    parser.add_argument('--tr_num', type=int, default=100,
                        help='Number of train trajectories.')
    parser.add_argument('--va_num', type=int, default=30,
                        help='Number of validation trajectories.')
    parser.add_argument('--te_num', type=int, default=30,
                        help='Number of test trajectories.')
    parser.add_argument('--exp_num', type=int, default=1, help='Number of repeated experiments')
    parser.add_argument(
        "--length", type=int, default=51000, help="Length of trajectory."
    )
    parser.add_argument('--sample_freq', type=int, default=100,
                        help='How often to sample the trajectory.')
    parser.add_argument('--interaction_strength', type=int, default=1,
                        help='Strength of Interactions between particles')

    args = parser.parse_args()
    return args

def generate_dataset(A, num_sims, length, sample_freq):
    num_sims = num_sims
    num_timesteps = int((length / float(sample_freq)) - 1)

    t0, t1, dt = 0, int((length / float(sample_freq)) / 10), 0.01
    T = np.arange(t0, t1, dt)

    sim_data_all = []
    edges_all = []
    for i in range(num_sims):
        print(f'Simulating training trajectory:{i+1:3d}  /{num_sims:3d}')
        t = time.time()

        sim_data = kuramoto.simulate_kuramoto(
            A, args.num_nodes, num_timesteps, T, dt
        )

        sim_data_all.append(sim_data)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

    
    data_all = np.array(sim_data_all, dtype=np.float32)
    #output data has shape [batch, time, nodes, variables]
    data_all = data_all.transpose(0,2,1,3)
    return data_all


if __name__ == "__main__":
    args = parse_args()
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
        print("Generating {} training simulations".format(args.tr_num))
        x_tr = generate_dataset(
            A, args.tr_num, args.length, args.sample_freq
        )

        print("Generating {} validation simulations".format(args.va_num))
        x_va = generate_dataset(
            A, args.va_num, args.length, args.sample_freq
        )

        print("Generating {} test simulations".format(args.te_num))
        x_te = generate_dataset(
            A, args.te_num, args.length, args.sample_freq
        )

        result = [x_tr,x_va,x_te,A]
        data_path = 'Kuramoto_' + args.graph + str(args.num_nodes) + '_exp' + str(exp_id) +'.pickle'
        with open(data_path, 'wb') as f:
            pickle.dump(result, f)