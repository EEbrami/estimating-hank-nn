import torch


# %% Layer to normalize the inputs
class NormalizeLayer(torch.nn.Module):
    def __init__(self, lower_bound, upper_bound):
        super(NormalizeLayer, self).__init__()

        # Register the lower bound and upper bound as buffers
        self.register_buffer("lower_bound", lower_bound)
        self.register_buffer("upper_bound", upper_bound)

    def forward(self, x):
        return 2 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1


# %% Set Transformer for permutation-invariant processing
class SetTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SetTransformer, self).__init__()
        
        # Phi network: processes each element independently
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.CELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.CELU()
        )
        
        # Rho network: processes the aggregated representation
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.CELU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, set_size, input_dim)
        
        # Process each element: (batch_size, set_size, hidden_dim)
        h = self.phi(x)
        
        # Aggregate (mean pooling): (batch_size, hidden_dim)
        h_agg = torch.mean(h, dim=1)
        
        # Process aggregate: (batch_size, output_dim)
        out = self.rho(h_agg)
        
        return out
