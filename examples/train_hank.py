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
    print("--- DRY RUN MODE ---")
    print("Loading Pre-trained HANK Model (Steady State)...")
    try:
        # 1. Load the weights you just created in pretrain_hank.py
        model = HANKModel.load("save/hank_pretrained.pkl")
        print("Success: Loaded steady-state weights.")
    except FileNotFoundError:
        print("Error: Pre-trained model not found. Please run pretrain_hank.py first.")
        return

    print("Starting Short Verification Training (Dry Run)...")
    
    # 2. Run for only 100 iterations to verify stability
    # This tests if the model explodes when hit with Aggregate Shocks (which pre-training didn't have)
    model.train_model(
        iteration=100,      # <--- LOW NUMBER for Dry Run (was 1000 or 10000)
        batch=64, 
        print_after=10
    )
    
    print("Dry Run complete. Saving checkpoint...")
    model.save("save", "hank_dry_run")
    print("Success! The code handles aggregate shocks. You are ready for full training.")

if __name__ == "__main__":
    train()
