import sys
import os
import pickle
import pandas as pd
import torch
import torch.nn as nn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from estimating_hank_nn.hank import HANKModel

# Redefine HANKModelMLP class to allow unpickling
class FlattenLayer(nn.Module):
    def __init__(self, start_dim=1):
        super(FlattenLayer, self).__init__()
        self.start_dim = start_dim
    def forward(self, x):
        return x.flatten(start_dim=self.start_dim)

class HANKModelMLP(HANKModel):
    def make_network(self, N_states=1, N_par=None, N_outputs=2, hidden=128, layers=4, activation=torch.nn.CELU(), normalize=True):
        if N_par is None:
            N_par = len(self.par.keys())
        self.distribution_encoder = FlattenLayer(start_dim=1)
        num_agents = 1000 
        dist_dim = num_agents * 3
        input_dim = N_states + N_par + dist_dim
        layers_list = []
        layers_list.append(nn.Linear(input_dim, hidden))
        layers_list.append(activation)
        for _ in range(layers - 1):
            layers_list.append(nn.Linear(hidden, hidden))
            layers_list.append(activation)
        layers_list.append(nn.Linear(hidden, N_outputs))
        self.network = nn.Sequential(*layers_list)

def reconstruct():
    print("Reconstructing full loss history...")
    # Path to the final model (or latest checkpoint)
    model_path = "save_mlp/hank_mlp_final.pkl"
    
    # Check if exists
    if not os.path.exists(model_path):
        print(f"Error: File not found: {model_path}")
        # Try to find the latest checkpoint
        checkpoints = [f for f in os.listdir("save_mlp") if f.startswith("hank_checkpoint_") and f.endswith(".pkl")]
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1]
            model_path = os.path.join("save_mlp", latest)
            print(f"Using latest checkpoint instead: {model_path}")
        else:
            return

    try:
        with open(model_path, "rb") as f:
            # We need to make sure pickle can find HANKModelMLP.
            # Since we defined it in __main__, we might need to map it.
            # But wait, the pickle was saved from 'train_mlp_baseline.py'.
            # So it expects 'train_mlp_baseline.HANKModelMLP'.
            # We can try to inject it into sys.modules or use a custom unpickler.
            # Or simpler: Just import train_mlp_baseline if possible?
            # No, that runs the training.
            
            # Hack: Map the module name
            import types
            mod = types.ModuleType("train_mlp_baseline")
            mod.HANKModelMLP = HANKModelMLP
            mod.FlattenLayer = FlattenLayer
            sys.modules["train_mlp_baseline"] = mod
            
            model = pickle.load(f)
            
            loss_history = model.loss_dict["total"]
            iterations = model.loss_dict["iteration"]
            
            df = pd.DataFrame({"iteration": iterations, "loss": loss_history})
            output_path = "save_mlp/loss_history_full.csv"
            df.to_csv(output_path, index=False)
            print(f"Success! Saved full history ({len(df)} steps) to {output_path}")
            
    except Exception as e:
        print(f"Error unpickling: {e}")
        # Fallback: Try to load the checkpoints if final fails?
        # But final exists. The issue is likely the class definition path.

if __name__ == "__main__":
    reconstruct()
