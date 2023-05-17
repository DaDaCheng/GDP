#implemented based on https://github.com/ethanfetaya/nri

import time
import numpy as np
import argparse
import networkx as nx
import pickle
from tqdm import tqdm
import multiprocessing



parser = argparse.ArgumentParser('Generate spring model simulation data')
parser.add_argument('--graph', type=str, default='BA')
parser.add_argument('--num-nodes', type=int, default=50,
                    help='Number of balls in the simulation.')
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


parser.add_argument('--length', type=int, default=501000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=501000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')

args = parser.parse_args()


class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, noise_var=0., graph = None):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

        self.edges = graph

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=10,seed=None):
        if seed is not None:
            np.random.seed(seed)
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0

        edges = self.edges
               
        # edges = np.tril(edges) + np.tril(edges, -1).T
        # np.fill_diagonal(edges, 0)

        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel


def compute_task(inputs):
    sim,length,sample_freq,seed=inputs
    result = sim.sample_trajectory(T=length,sample_freq=sample_freq,seed=seed)
    return result
    

def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()    
    data = [(sim, length,sample_freq,i) for i in range(num_sims)]
    pool = multiprocessing.Pool()
    result_iter = pool.imap(compute_task, data)
    result_list = []
    for result in tqdm(result_iter, total=len(data)):
        loc_all.append(result[0])
        vel_all.append(result[1])
    # Close the pool
    pool.close()
    # Join the pool
    pool.join()
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)

    return loc_all, vel_all

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

        sim = SpringSim(noise_var=0.0, n_balls=n, graph = A)
        suffix = '_springs_'
        suffix += args.graph
        suffix += str(n)
        print("Generating {} training simulations".format(args.tr_num))
        loc_train, vel_train = generate_dataset(args.tr_num,
                                                             args.length,
                                                             args.sample_freq)

        print("Generating {} validation simulations".format(args.va_num))
        loc_valid, vel_valid = generate_dataset(args.va_num,
                                                             args.length,
                                                             args.sample_freq)

        print("Generating {} test simulations".format(args.te_num))
        loc_test, vel_test = generate_dataset(args.te_num,
                                                          args.length_test,
                                                          args.sample_freq)

        loc_train, vel_train = loc_train.astype(np.float16), vel_train.astype(np.float16)
        loc_valid, vel_valid = loc_valid.astype(np.float16), vel_valid.astype(np.float16)
        loc_test, vel_test = loc_test.astype(np.float16), vel_test.astype(np.float16)


        # save data, output data has shape [batch, time, nodes, variables]
        x_tr = np.concatenate((loc_train,vel_train),axis=2)
        x_tr = x_tr.transpose(0,1,3,2)

        x_va = np.concatenate((loc_valid,vel_valid),axis=2)
        x_va = x_va.transpose(0,1,3,2)

        x_te = np.concatenate((loc_test,vel_test),axis=2)
        x_te = x_te.transpose(0,1,3,2)

        A = sim.edges

        result = [x_tr,x_va,x_te,A]
        data_path = 'Spring_' + args.graph + str(n) + '_exp' + str(exp_id) +'.pickle'

        with open(data_path, 'wb') as f:
            pickle.dump(result, f)
