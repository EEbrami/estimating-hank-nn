import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def plot_losses(csv_path="save/loss_history.csv", output_dir="save"):
    if not os.path.exists(csv_path):
        print(f"Error: Loss history file not found at {csv_path}")
        return

    print(f"Loading loss history from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot 1: Total Loss (Log Scale)
    plt.figure(figsize=(10, 6))
    plt.plot(df["iteration"], df["total"], label="Total Loss", color="black")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Training Convergence: Total Loss")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig(f"{output_dir}/loss_total.png")
    print(f"Saved {output_dir}/loss_total.png")

    # Plot 2: Component Losses
    plt.figure(figsize=(10, 6))
    plt.plot(df["iteration"], df["nkpc"], label="NKPC Loss", alpha=0.7)
    plt.plot(df["iteration"], df["bond_euler"], label="Euler Loss", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Training Convergence: Component Losses")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig(f"{output_dir}/loss_components.png")
    print(f"Saved {output_dir}/loss_components.png")

if __name__ == "__main__":
    # Allow passing a different path via command line
    path = "save/loss_history.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    plot_losses(path)
