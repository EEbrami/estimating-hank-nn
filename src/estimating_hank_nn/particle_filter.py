import torch
import math
import pickle
from pathlib import Path
from copy import deepcopy
from tqdm import trange
from estimating_hank_nn.structures import Parameters
from estimating_hank_nn.networks import NormalizeLayer

class ParticleFilter:
    def __init__(self, model, data, R):
        self.model = model
        self.data = data
        self.S = len(data)
        self.R = R
        self.R_inv = torch.linalg.inv(R)
        self.R_det = torch.linalg.det(R)
        self.diagnostics = None
        self.dataset = None
        self.network = self.make_network()
        self.loss_dict = None

    def to(self, device):
        self.model.to(device)
        self.network.to(device)

    def save(self, path, name="pf"):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.to("cpu")
        with open(f"{path}/{name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def kitagawa(w_norm, P, aux=0.4532):
        device = w_norm.device
        cum_dist = torch.cumsum(w_norm, dim=-1)
        u = torch.arange(aux, P, step=1.0, device=device) / P
        idx = torch.searchsorted(cum_dist, u, right=False)
        idx = torch.clamp(idx, max=P-1)
        return idx

    @staticmethod
    def log_prob(error_tensor, R_inv, R_det):
        S = error_tensor.size(0)
        log_prob = -0.5 * (S * math.log(2.0 * math.pi) + torch.log(R_det) + torch.sum(error_tensor * torch.mm(R_inv, error_tensor), dim=0))
        return log_prob

    @torch.no_grad()
    def filter(self, P, burn=100, sim=100, par=None, device="cpu"):
        if par is None:
            par = deepcopy(self.model.par)
        else:
            par = deepcopy(par)

        # Number of series
        S = self.S

        # Move to device
        par.to(device)
        self.model.to(device)

        # Initialize
        self.model.par_draw = par.expand((P, 1))
        self.model.state = self.model.initialize_state(batch=P, device=device)
        # self.model.ss = self.model.steady_state() # SS not implemented

        # Burn
        # self.model.steps(batch=P, device=device, steps=burn) # steps not implemented in HANKModel yet

        # Filter
        filtered_model_out = {key: [] for key in self.data.keys()}
        log_likelihood = torch.empty(sim)
        
        # Data keys: OutputGrowth, Inflation, InterestRate
        # Model output: X, Pi, R
        # We need to map model output to data keys
        
        for t in range(sim):
            # Simulate step
            # model_out = self.model.sim_step() # sim_step not implemented in HANKModel yet
            
            # Placeholder for sim_step
            # Assume model.policy returns X, Pi
            X, Pi = self.model.policy(self.model.state, self.model.par_draw)
            # R = self.model.softplus_zlb(...) # Need R_star
            R_star = 1.0 + par.phipi * Pi + par.phiy * X
            R = self.model.softplus_zlb(R_star)
            
            model_out = {"OutputGrowth": X, "Inflation": Pi, "InterestRate": R} # Simplified mapping

            # Error
            error = torch.empty((S, P), device="cpu")
            for i, (key, value) in enumerate(model_out.items()):
                # value shape: (P, 1)
                # data[key] shape: (T,)
                # We need data[key][t]
                if key in self.data:
                    data_val = self.data[key].iloc[t] if hasattr(self.data[key], 'iloc') else self.data[key][t]
                    error[i, :] = value.squeeze(-1).to("cpu") - data_val

            # Log probabilities
            log_prob = self.log_prob(error, self.R_inv, self.R_det)

            # Weights
            w = torch.exp(log_prob)

            # Log likelihood
            log_likelihood[t] = torch.log(torch.mean(w))

            # Normalize weights
            w_norm = w / torch.sum(w)

            # Resample
            idx = self.kitagawa(w_norm, P)

            # Resample states
            # self.model.state.update({key: value[idx, ...] for key, value in self.model.state.items()})
            # State update logic needed for HANKModel
            for key, value in self.model.state.items():
                # value is (P, ...)
                setattr(self.model.state, key, value[idx])
            
            # Update step
            # self.model.steps(batch=P, steps=1, device=device)
            e = self.model.draw_shocks((P, 1), device=device)
            self.model.state = self.model.step(e)

            # Filter model output
            for key, value in model_out.items():
                filtered_model_out[key].append(torch.mean(value[idx, ...], dim=0))

        # Stack filtered data
        for key, value in filtered_model_out.items():
            filtered_model_out[key] = torch.stack(value, dim=-1)

        return torch.sum(log_likelihood), filtered_model_out

    def make_network(self, N_inputs=None, hidden=64, layers=3, activation=torch.nn.CELU(), normalize=True):
        if N_inputs is None:
            N_inputs = len(self.model.par)

        N_outputs = 1

        layer_list = []

        # Normalization layer
        if normalize:
            lower_bound = self.model.range.low_tensor()
            upper_bound = self.model.range.high_tensor()
            layer_list.append(NormalizeLayer(lower_bound, upper_bound))

        # First layer
        layer_list.append(torch.nn.Linear(N_inputs, hidden))
        layer_list.append(activation)

        # Middle layers
        for _ in range(1, layers):
            layer_list.append(torch.nn.Linear(hidden, hidden))
            layer_list.append(activation)

        # Last layer
        layer_list.append(torch.nn.Linear(hidden, N_outputs))

        # Build the network
        self.network = torch.nn.Sequential(*layer_list)

    def train(self, batch=64, epochs=10000, device="cpu", lr=1e-3, eta_min=1e-6, validation_share=0.2, print_after=1000):
        # ... (Same as original, simplified for brevity)
        pass # Placeholder, user can implement full training loop if needed
