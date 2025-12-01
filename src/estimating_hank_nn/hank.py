import torch
import pickle
from pathlib import Path
from estimating_hank_nn.structures import Parameters, State, Ranges, Shocks
from estimating_hank_nn.networks import NormalizeLayer, SetTransformer
from estimating_hank_nn.helpers import ergodic_sigma

class HANKModel(object):
    def __init__(self, parameters, ranges, shocks) -> None:
        self.range = Ranges(parameters, ranges)
        self.shock = Shocks(shocks)
        self.par = Parameters(parameters)
        self.par_draw = None
        self.ss = None
        self.state = None
        self.network = self.make_network()
        self.loss_dict = None
        self.training_conf = None

    def steady_state(self, par=None):
        if par is None:
            par = self.par_draw

        # Calculate derived parameters for HANK (simplified)
        # In a full HANK, this would solve for the stationary distribution.
        # Here we return parameters useful for the aggregate equations.
        
        # Kappa (slope of NKPC) - simplified RANK-like formula or calibrated
        kappa = ((1 - par.phi) * (1 - par.phi * par.beta) * (par.sigma + par.eta)) / par.phi
        
        # Omega (TFP shock scaling)
        omega = (1 + par.eta) / (par.sigma + par.eta)

        return Parameters({"kappa": kappa, "omega": omega})

    def to(self, device):
        self.par.to(device)
        self.par_draw.to(device)
        # self.ss.to(device) # SS might be complex in HANK
        self.state.to(device)
        self.network.to(device)

    def save(self, path, name="model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.to("cpu")
        with open(f"{path}/{name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def make_network(self, N_states=1, N_par=None, N_outputs=2, hidden=64, layers=5, activation=torch.nn.CELU(), normalize=True):
        # Detect device
        device = self.par.values()[0].device

        # Number of parameters
        if N_par is None:
            N_par = len(self.par)
        
        # HANK Input: Aggregate State (Zeta) + Parameters + Distribution (via SetTransformer)
        # For now, we assume the distribution is handled by a separate branch or implicitly
        # But to follow the plan, we should integrate SetTransformer.
        
        # Let's assume the main network takes: [AggregateState, Parameters, DistributionEmbedding]
        # DistributionEmbedding comes from SetTransformer.
        
        # Dimensions
        dist_input_dim = 3 # (b, a, e)
        dist_embedding_dim = 16
        
        self.distribution_encoder = SetTransformer(dist_input_dim, hidden, dist_embedding_dim)
        
        N_inputs = N_states + N_par + dist_embedding_dim

        layer_list = []

        # Normalize layer (only for scalar inputs, distribution is handled separately)
        # We might need a custom normalize layer that handles the concatenation
        # For simplicity, we'll skip normalization for the embedding part or normalize after concat
        
        # First layer
        layer_list.append(torch.nn.Linear(N_inputs, hidden))
        layer_list.append(activation)

        # Middle layers
        for _ in range(1, layers):
            layer_list.append(torch.nn.Linear(hidden, hidden))
            layer_list.append(activation)

        # Last layer
        layer_list.append(torch.nn.Linear(hidden, N_outputs))

        return torch.nn.Sequential(*layer_list)

    def initialize_state(self, par=None, batch=100, multiplier=1.0, device="cpu"):
        # Initialize aggregate state (TFP shock)
        if par is None:
            par = self.par_draw
            
        # Steady state
        ss = self.steady_state(par=par)
        
        # Ergodic standard deviation of zeta
        # sigma_z = sigma_a / sqrt(1 - rho_a^2)
        # But in the code it was: sigma = par.sigma_a * par.sigma * (par.rho_a - 1) * ss.omega
        # Let's stick to the simple AR(1) ergodic std for zeta
        ergodic = ergodic_sigma(par.rho_a, par.sigma_a)
            
        zeta = torch.randn((batch, 1), device=device) * ergodic * multiplier
        
        # Initialize distribution (random for now, should be steady state)
        # Shape: (batch, num_agents, 3) -> (b, a, e)
        num_agents = 1000
        # For now, initialize with some random heterogeneity
        dist = torch.randn((batch, num_agents, 3), device=device)
        
        return State({"zeta": zeta, "distribution": dist})

    def draw_parameters(self, shape, device="cpu"):
        return self.range.sample(shape, device=device)

    def draw_shocks(self, shape, antithetic=False, device="cpu"):
        return self.shock.sample(shape, antithetic, device=device)

    def policy(self, state=None, par=None):
        if state is None:
            state = self.state
        if par is None:
            par = self.par_draw

        # Encode distribution
        dist_embedding = self.distribution_encoder(state.distribution)
        
        # Vector of states and parameters
        input_state = state.zeta
        input_par = par.cat()

        # Expand if necessary
        if input_state.ndim > input_par.ndim:
             input_par = input_par.expand(input_state.size(0), -1)

        # Concatenate
        input = torch.cat([input_state, input_par, dist_embedding], dim=-1)

        # Evaluate network
        output = self.network(input)

        # Output: X (Output Gap), Pi (Inflation) - simplified
        X = output[..., 0:1] / 100
        Pi = output[..., 1:2] / 100

        return X, Pi

    def softplus_zlb(self, R_star, kappa=10.0):
        """
        Smooth approximation of max(1, R_star) using Softplus.
        R = 1 + (1/kappa) * log(1 + exp(kappa * (R_star - 1)))
        """
        return 1.0 + (1.0 / kappa) * torch.log(1.0 + torch.exp(kappa * (R_star - 1.0)))

    def fischer_burmeister(self, a, b):
        """
        Fischer-Burmeister function for complementarity problems.
        phi(a, b) = sqrt(a^2 + b^2) - (a + b) = 0  <=>  a >= 0, b >= 0, ab = 0
        """
        return torch.sqrt(a**2 + b**2) - (a + b)

    def residuals(self, e):
        par = self.par_draw
        # ss = self.ss # Steady state might be needed for some parameters
        state = self.state

        # 1. Policy Evaluation at t
        X, Pi = self.policy(state, par)
        
        # 2. Next Period State
        # Use the step function to evolve state (aggregate + distribution)
        state_next = self.step(e)
        
        # 3. Policy Evaluation at t+1 (Expectations)
        X_next, Pi_next = self.policy(state_next, par)
        EX_next = torch.mean(X_next, dim=0)
        EPi_next = torch.mean(Pi_next, dim=0)
        
        # 4. Equilibrium Conditions
        
        # Taylor Rule with ZLB
        # R_t = max(1, R_star * (Pi/Pi_star)^phi_pi * (Y/Y_star)^phi_y)
        # Log-linear: r_t = max(0, r_star + phi_pi * pi + phi_y * y)
        # We use Softplus for differentiability
        R_star_t = 1.0 + par.phipi * Pi + par.phiy * X 
        R_t = self.softplus_zlb(R_star_t)
        
        # NKPC (New Keynesian Phillips Curve)
        # pi_t = kappa * x_t + beta * E_t[pi_{t+1}]
        nkpc = Pi - (par.kappa * X + par.beta * EPi_next)
        
        # Aggregate Euler Equation (IS Curve)
        # Note: In HANK, this is an approximation. The "Investment Wedge" or "Discount Factor Wedge" 
        # would appear here if we derived it from the aggregated individual Euler equations.
        # By training the network to satisfy this *Aggregate* relation, we are effectively 
        # finding the HANK equilibrium that mimics RANK aggregates but is conditioned on the distribution.
        # Ideally, we would minimize the aggregation of individual Euler errors, but this is a reasonable proxy
        # for the "Extension" scope.
        bond_euler = X - (EX_next - 1 / par.sigma * (R_t - EPi_next - state.zeta))

        return torch.sum(nkpc**2), torch.sum(bond_euler**2)

    @torch.no_grad()
    def step(self, e):
        par = self.par_draw
        # ss = self.ss
        state = self.state

        # Update aggregate state (TFP shock)
        # zeta_next = rho * zeta + sigma_a * epsilon
        zeta_next = par.rho_a * state.zeta + e.zeta * par.sigma_a
        
        # Update distribution
        # In a full model, we would simulate agents: b' = g(b, e, zeta, ...)
        # For now, we assume a stationary distribution or simple evolution
        # Placeholder: keep distribution constant (or add small noise to simulate churn)
        dist_next = state.distribution # + 0.01 * torch.randn_like(state.distribution)

        return State({"zeta": zeta_next, "distribution": dist_next})

    def steps(self, batch, device, steps):
        # Initialize
        state = self.initialize_state(batch=batch, device=device)
        self.state = state
        
        # Simulate
        for _ in range(steps):
            e = self.draw_shocks((batch, 1), device=device)
            self.state = self.step(e)

    def train_model(
        self,
        iteration=10000,
        internal=1,
        steps=10,
        batch=100,
        mc=10,
        par_draw_after=100,
        lr=1e-3,
        eta_min=1e-10,
        device="cpu",
        print_after=100,
    ):
        # Save training configuration
        self.training_conf = locals().copy()

        # Print training configuration
        print("Training configuration:")
        for key, value in self.training_conf.items():
            if key != "self":
                print(f"{key}: {value}")

        # Set the network to train mode
        self.network.train()
        self.network.to(device)

        # Initialize
        self.par_draw = self.draw_parameters(shape=(batch, 1), device=device)
        self.state = self.initialize_state(batch=batch, device=device)
        # self.ss = self.steady_state() # SS not implemented yet

        # Starting weights for loss components
        weights = [1.0, 1.0]

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iteration, eta_min=eta_min)

        # Dictionary for loss
        self.loss_dict = {"iteration": [], "total": [], "nkpc": [], "bond_euler": []}

        # Training loop
        running_loss = 0.0
        for i in range(iteration):
            for o in range(internal):
                optimizer.zero_grad()
                e = self.draw_shocks((mc, batch, 1), antithetic=True, device=device)
                
                loss_nkpc, loss_euler = self.residuals(e)
                
                # Normalize by batch and mc
                loss_nkpc = loss_nkpc / (batch * mc)
                loss_euler = loss_euler / (batch * mc)
                
                loss = weights[0] * loss_nkpc + weights[1] * loss_euler
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

                # Record loss
                if o == 0:
                    self.loss_dict["iteration"].append(i)
                    self.loss_dict["total"].append(loss.item())
                    self.loss_dict["nkpc"].append(loss_nkpc.item())
                    self.loss_dict["bond_euler"].append(loss_euler.item())

                    # Running loss
                    running_loss += loss.item()

            # Print running loss
            if i % print_after == 0:
                print(f"Iteration {i}, Loss: {running_loss / print_after:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0

            # Update learning rate
            scheduler.step()

            # Draw new parameters
            if i % par_draw_after == 0:
                self.par_draw = self.draw_parameters((batch, 1), device=device)
                # self.ss = self.steady_state()

            # Sample states by simulation (optional, or just re-initialize)
            # self.steps(batch=batch, device=device, steps=steps) 
            # For now, let's just re-initialize state occasionally or evolve it
            self.state = self.initialize_state(batch=batch, device=device)

        # Set network to evaluation mode
        self.network.eval()

