import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from estimating_hank_nn.hank import HANKModel
from estimating_hank_nn.networks import NormalizeLayer

# %% Flatten Layer (The "Naive" Approach)
class FlattenLayer(nn.Module):
    def __init__(self, start_dim=1):
        super(FlattenLayer, self).__init__()
        self.start_dim = start_dim

    def forward(self, x):
        return x.flatten(start_dim=self.start_dim)

# %% MLP Baseline Model
class HANKModelMLP(HANKModel):
    def make_network(self, N_states=1, N_par=None, N_outputs=2, hidden=128, layers=4, activation=torch.nn.CELU(), normalize=True):
        if N_par is None:
            N_par = len(self.par.keys())
            
        # 1. Distribution Encoder: Just flatten it!
        # Input: (batch, agents, 3) -> Output: (batch, agents*3)
        self.distribution_encoder = FlattenLayer(start_dim=1)
        
        # 2. Main Network (MLP)
        # Input: zeta (1) + parameters (N_par) + distribution (1000 * 3)
        # Note: We assume 1000 agents. If this changes, the model breaks (another downside of MLPs!)
        num_agents = 1000 
        dist_dim = num_agents * 3
        input_dim = N_states + N_par + dist_dim
        
        layers_list = []
        
        # Normalization (Optional but recommended)
        if normalize:
            # We can't easily normalize the flattened distribution with a simple layer 
            # without knowing bounds for every agent slot. 
            # So we skip normalization for the distribution part or use a BatchNorm.
            # Let's use a simple Linear layer first.
            pass

        # Input Layer
        layers_list.append(nn.Linear(input_dim, hidden))
        layers_list.append(activation)
        
        # Hidden Layers
        for _ in range(layers - 1):
            layers_list.append(nn.Linear(hidden, hidden))
            layers_list.append(activation)
            
        # Output Layer
        layers_list.append(nn.Linear(hidden, N_outputs))
        
        self.network = nn.Sequential(*layers_list)
        
        # Move to device handled by train_model or explicit to() call
        # self.to(self.device)

# %% Training Loop (Copied and adapted from train_hank.py)
# Configuration (Same as train_hank.py)
NK_par = {
    "rho_a": 0.95,
    "sigma_a": 0.007,
    "sigma": 2.0,
    "phi": 100.0,
    "phipi": 1.5,
    "phiy": 0.125,
    "beta": 0.99,
    "eta": 1.0,
    "kappa": 0.1, # Derived or fixed
    "omega": 1.0  # Derived or fixed
}

NK_range = {
    "rho_a": torch.distributions.Uniform(0.90, 0.98),
    "sigma_a": torch.distributions.Uniform(0.005, 0.010),
    "sigma": torch.distributions.Uniform(1.5, 2.5),
    "phi": torch.distributions.Uniform(50.0, 150.0),
    "phipi": torch.distributions.Uniform(1.1, 2.0),
    "phiy": torch.distributions.Uniform(0.05, 0.20),
    "beta": torch.distributions.Uniform(0.98, 0.995),
    "eta": torch.distributions.Uniform(0.5, 1.5)
}

shock_dist = {
    "zeta": torch.distributions.Normal(0, 1),
    "m_shock": torch.distributions.Normal(0, 1)
}

def train():
    print("--- MLP BASELINE EXPERIMENT (50k Steps) ---")
    print("Initializing HANKModelMLP (Naive Flattening)...")
    
    # Initialize Model
    model = HANKModelMLP(NK_par, NK_range, shock_dist)
    
    # Create the network (MLP)
    # Input: 1 (zeta) + 9 (params) + 3000 (dist) = 3010
    model.make_network(hidden=128, layers=4)
    
    print("Starting Training...")
    
    # Train
    model.train_model(
        iteration=100000, # <--- INCREASED TO 100K
        internal=1,
        steps=10,
        batch=64, 
        mc=10,
        par_draw_after=100,
        lr=1e-3,
        eta_min=1e-10,
        device="cpu",
        print_after=100,
        save_every=1000,
        save_path="save_mlp" # Separate save directory
    )
    
    print("MLP Baseline Training Complete.")
    model.save("save_mlp", "hank_mlp_final")
    print("Saved to save_mlp/hank_mlp_final.pkl")

if __name__ == "__main__":
    # Create save directory
    if not os.path.exists("save_mlp"):
        os.makedirs("save_mlp")
    train()
