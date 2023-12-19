# GDP


## About

This is the source code for paper _A Dynamical Graph Prior for Relational Inference_.

## Requirements

python>=3.3.7

torch>=1.9.0

torch-cluster>=1.5.9

torch-geometric>=2.0.0

torch-scatter>=2.0.8

torch-sparse>=0.6.11

tqdm

This code was tested on macOS and Linux.

## Run

### Quick start

	python train_DYGR.py --suffix MM_ER50_exp0

### Available Datasets

1.  Michaelis–Menten kinetics, a model for gene regulation circuits.

2.  Rössler oscillators on graphs, which generate chaotic dynamics.

3.  Diffusion, a continuous-time linear dynamics.

4.  Spring model that describes particles connected by springs and interacts via Hooke’s law;

5.  Kuramoto model that describes phase-coupled oscillators placed on a graph.

6.  Friedkin-Johnsen dynamics, a classical model for describing opinion formation, polarization and filter bubble in social networks;

7.  The coupled map network, a discrete-time model with chaotic behavior. 

8.  Netsim, a simulated fMRI data. 


