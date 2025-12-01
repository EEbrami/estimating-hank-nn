import torch
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
    "kappa": 0.1,
}

NK_range = {
    "beta": torch.distributions.Uniform(0.95, 0.99),
    "sigma": torch.distributions.Uniform(1.0, 3.0),
    "eta": torch.distributions.Uniform(0.25, 2.0),
    "phi": torch.distributions.Uniform(0.5, 0.9),
    "phipi": torch.distributions.Uniform(1.25, 2.5),
    "phiy": torch.distributions.Uniform(0.0, 0.5),
    "rho_a": torch.distributions.Uniform(0.8, 0.95),
    "sigma_a": torch.distributions.Uniform(0.02, 0.1),
}

shock_dist = {
    "zeta": torch.distributions.Normal(0.0, 1.0),
}

def train():
    print("Initializing HANK Model...")
    model = HANKModel(NK_par, NK_range, shock_dist)
    
    print("Starting Training...")
    # Train for a small number of iterations for demonstration
    model.train_model(iteration=1000, batch=64, print_after=100)
    
    print("Training complete.")
    model.save("save", "hank_trained")

if __name__ == "__main__":
    train()
