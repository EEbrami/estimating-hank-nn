import torch
import pandas as pd
from estimating_hank_nn.hank import HANKModel
from estimating_hank_nn.particle_filter import ParticleFilter
from estimating_hank_nn.structures import Parameters
import sys
from pathlib import Path

# Add data directory to path to import process_data
sys.path.append(str(Path(__file__).parent.parent / "data"))
try:
    from process_data import load_data
except ImportError:
    # Fallback if running from examples dir
    sys.path.append("../data")
    from process_data import load_data

# Configuration (Same as train_hank.py)
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

def estimate():
    print("Loading Data...")
    df = load_data()
    # Convert to dictionary of tensors
    data = {
        "OutputGrowth": torch.tensor(df["OutputGrowth"].values, dtype=torch.float32),
        "Inflation": torch.tensor(df["Inflation"].values, dtype=torch.float32),
        "InterestRate": torch.tensor(df["InterestRate"].values, dtype=torch.float32)
    }
    
    # Measurement error covariance (diagonal)
    R_diag = torch.tensor([0.1, 0.1, 0.1]) # Assumed measurement error variance
    R = torch.diag(R_diag)
    
    print("Initializing HANK Model...")
    model = HANKModel(NK_par, NK_range, shock_dist)
    
    # Load trained model if available
    try:
        model = HANKModel.load("save/hank_pretrained.pkl")
        print("Loaded trained model.")
    except FileNotFoundError:
        print("Trained model not found. Using initialized model (random weights).")
    
    print("Initializing Particle Filter...")
    pf = ParticleFilter(model, data, R)
    
    print("Calculating Likelihood at initial parameters...")
    # Run filter
    log_likelihood, filtered_data = pf.filter(P=50, sim=len(df))
    
    print(f"Log Likelihood: {log_likelihood.item():.4f}")
    
    # Simple Grid Search for Sigma
    print("\nRunning Grid Search for Sigma...")
    sigmas = torch.linspace(0.5, 5.0, 10)
    likelihoods = []
    
    for s in sigmas:
        par = Parameters(NK_par)
        par.sigma = s
        ll, _ = pf.filter(P=50, sim=len(df), par=par)
        likelihoods.append(ll.item())
        print(f"Sigma: {s:.2f}, LL: {ll.item():.4f}")
        
    # Save results
    results = pd.DataFrame({"Sigma": sigmas.numpy(), "LogLikelihood": likelihoods})
    results.to_csv("save/estimation_results.csv", index=False)
    print("Estimation results saved to save/estimation_results.csv")

if __name__ == "__main__":
    estimate()
