import torch
import numpy as np
from estimating_hank_nn.hank import HANKModel
from estimating_hank_nn.structures import Parameters

# Configuration
NK_par = {
    "beta": 0.97,
    "sigma": 2.0,
    "eta": 1.125,
    "phi": 0.7,
    "phipi": 1.875,
    "phiy": 0.25,
    "rho_a": 0.875,
    "sigma_a": 0.06,
    "kappa": 0.1, # Added kappa explicitly
}

NK_range = {
    "beta": torch.distributions.Uniform(0.95, 0.99),
    "sigma": torch.distributions.Uniform(1.0, 3.0),
}

shock_dist = {
    "zeta": torch.distributions.Normal(0.0, 1.0),
}

def pretrain():
    print("Initializing HANK Model...")
    model = HANKModel(NK_par, NK_range, shock_dist)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.network.parameters(), lr=1e-3)
    
    print("Starting Pre-training (Steady State)...")
    # In pre-training, we might want to train on the deterministic steady state
    # or just run a few epochs to ensure stability.
    
    for i in range(100):
        optimizer.zero_grad()
        
        # Draw parameters and state
        model.par_draw = model.draw_parameters((10, 1))
        model.state = model.initialize_state(batch=10)
        e = model.draw_shocks((10, 1))
        
        # Compute residuals
        loss_nkpc, loss_euler = model.residuals(e)
        loss = loss_nkpc + loss_euler
        
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.6f}")
            
    print("Pre-training complete.")
    model.save("save", "hank_pretrained")

if __name__ == "__main__":
    pretrain()
