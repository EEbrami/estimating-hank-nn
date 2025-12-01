import torch
from estimating_hank_nn.networks import SetTransformer

def check_architecture():
    print("Checking SetTransformer Architecture...")
    
    # Parameters
    input_dim = 3
    hidden_dim = 64
    output_dim = 16
    batch_size = 5
    set_size = 10
    
    # Initialize model
    model = SetTransformer(input_dim, hidden_dim, output_dim)
    print("Model initialized successfully.")
    
    # Create dummy input
    x = torch.randn(batch_size, set_size, input_dim)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    try:
        out = model(x)
        print(f"Output shape: {out.shape}")
        
        # Check output shape
        expected_shape = (batch_size, output_dim)
        if out.shape == expected_shape:
            print("SUCCESS: Output shape matches expected shape.")
        else:
            print(f"FAILURE: Expected {expected_shape}, got {out.shape}")
            
    except Exception as e:
        print(f"FAILURE: Forward pass failed with error: {e}")

if __name__ == "__main__":
    check_architecture()
