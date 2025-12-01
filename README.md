# Extending HANK with Neural Networks

This repository contains an extension of the KMR framework (Kase, Melosi, Rottner, 2022) to solve and estimate a **Heterogeneous Agent New Keynesian (HANK)** model using Neural Networks.

While the original repository provided a RANK example, this branch implements the full HANK model with a **Set Transformer** to handle the infinite-dimensional distribution of agents.

## Key Features

- **HANK Model**: A One-Asset HANK model with heterogeneous households subject to uninsurable income risk and borrowing constraints.
- **Set Transformer**: A permutation-invariant neural network architecture using **Multihead Attention** (MAB/ISAB) to embed the agent distribution $\Gamma_t$ into a fixed-size vector.
- **Non-Linear Solution**: Solves for global policy functions, capturing the interaction between inequality and the Zero Lower Bound (ZLB).
- **Wide Net Strategy**: Trains on a broad range of parameters ($\sigma \in [0.5, 5.0]$) to ensure robustness and avoid extrapolation.
- **Monetary Shocks**: Includes monetary policy disturbances to capture interest rate volatility.
- **Neural Network Particle Filter**: A likelihood-based estimation method for non-linear heterogeneous agent models.

## Installation

```bash
git clone -b set_transformer https://github.com/EEbrami/estimating-hank-nn.git
cd estimating-hank-nn
pip install .
```

## Usage

### 1. Verification (New!)

Before running long training jobs, verify the architecture and stability:

```bash
# 1. Check Set Transformer Architecture (Output Shape)
python check_arch.py

# 2. Run a Dry Run (100 iterations) to verify stability and checkpointing
python examples/train_hank.py
```

_See `dry_run_log_v2.md` for example output._

### 2. Pre-training

Train the model on the deterministic steady state to ensure stability.

```bash
python examples/pretrain_hank.py
```

### 3. Full Training

Train the model over the full state space (Aggregate Shocks + Distribution).

```bash
# Edit the script to set iteration=10000 and save_every=1000
python examples/train_hank.py
```

**Checkpointing**: The model now saves checkpoints (e.g., `save/hank_checkpoint_1000.pkl`) automatically. If the run crashes, you can load the latest checkpoint in the script.

### 4. Estimation

Run the Neural Network Particle Filter to estimate parameters using real US data.

```bash
python examples/estimate_hank.py
```

## File Structure

- `src/estimating_hank_nn/`
  - `hank.py`: The `HANKModel` class (The "Brain").
  - `networks.py`: The `SetTransformer` architecture (The "Eye") with **Attention**.
  - `particle_filter.py`: The estimation algorithm.
- `examples/`
  - `analytical.py`: The original RANK example (for reference).
  - `train_hank.py`: Main training loop for HANK.
  - `estimate_hank.py`: Estimation script.

## References

- Kase, H., Melosi, L., & Rottner, M. (2022). _Estimating Heterogeneous Agent Models with Neural Networks_.
- Kaplan, G., Moll, B., & Violante, G. L. (2018). _Monetary Policy According to HANK_.
- Lee, J., et al. (2019). _Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks_.
