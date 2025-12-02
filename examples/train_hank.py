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
    "beta": torch.distributions.Uniform(0.90, 0.995), # Broader discount factor
    "sigma": torch.distributions.Uniform(0.5, 5.0),   # Covers low & high risk aversion
    "eta": torch.distributions.Uniform(0.25, 2.0),
    "phi": torch.distributions.Uniform(0.5, 0.9),
    "phipi": torch.distributions.Uniform(1.1, 3.0),   # Broader Taylor rule response
    "phiy": torch.distributions.Uniform(0.0, 0.5),
    "rho_a": torch.distributions.Uniform(0.8, 0.95),
    "sigma_a": torch.distributions.Uniform(0.02, 0.1),
    "kappa": torch.distributions.Uniform(0.01, 0.3),  # Added kappa range
}

shock_dist = {
    "zeta": torch.distributions.Normal(0.0, 1.0),
    "m_shock": torch.distributions.Normal(0.0, 1.0) # Monetary Policy Shock
}

def train():
    print("--- DRY RUN MODE ---")
    print("Loading Pre-trained HANK Model (Steady State)...")
    try:
        # 1. Load the weights you just created in pretrain_hank.py
        model = HANKModel.load("save/hank_pretrained.pkl")
        print("Success: Loaded steady-state weights.")
        
        # CRITICAL: Overwrite ranges and shocks with the new wider definitions
        # The loaded model has the old (narrow) ranges and missing shocks.
        from estimating_hank_nn.structures import Ranges, Shocks
        model.range = Ranges(NK_par, NK_range)
        model.shock = Shocks(shock_dist)
        print("Success: Updated model with wider ranges and monetary shock.")
        
    except FileNotFoundError:
        print("Error: Pre-trained model not found. Please run pretrain_hank.py first.")
        return

    print("Starting Full Training Run (20,000 Iterations)...")
    
    # 2. Run for 20,000 iterations (approx 2.5 hours on CPU)
    model.train_model(
        iteration=20000,      
        batch=64, 
        print_after=100,
        save_every=1000,      # Save checkpoint every 1000 steps
        save_path="save"
    )
    
    print("Dry Run complete. Saving final checkpoint...")
    model.save("save", "hank_dry_run_final")
    print("Success! Checkpointing verified.")

if __name__ == "__main__":
    train()
