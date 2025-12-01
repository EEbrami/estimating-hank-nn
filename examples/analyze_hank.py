import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze():
    # Create figures directory
    Path("figures").mkdir(parents=True, exist_ok=True)
    
    # Load results
    try:
        df = pd.read_csv("save/estimation_results.csv")
    except FileNotFoundError:
        print("Results file not found. Run estimate_hank.py first.")
        return

    # Plot Likelihood
    plt.figure(figsize=(10, 6))
    plt.plot(df["Sigma"], df["LogLikelihood"], marker='o')
    plt.title("Log Likelihood vs Sigma (Relative Risk Aversion)")
    plt.xlabel("Sigma")
    plt.ylabel("Log Likelihood")
    plt.grid(True)
    
    # Save figure
    plt.savefig("figures/likelihood_sigma.png")
    print("Saved plot to figures/likelihood_sigma.png")

if __name__ == "__main__":
    analyze()
