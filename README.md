# Extending HANK with Neural Networks

This repository contains an extension of the KMR framework (Kase, Melosi, Rottner, 2022) to solve and estimate a **Heterogeneous Agent New Keynesian (HANK)** model using Neural Networks.

While the original repository provided a RANK example, this branch implements the full HANK model with a **Set Transformer** to handle the infinite-dimensional distribution of agents.

## Key Features

- **HANK Model**: A One-Asset HANK model with heterogeneous households subject to uninsurable income risk and borrowing constraints.
- **Set Transformer**: A permutation-invariant neural network architecture that embeds the agent distribution $\Gamma_t$ into a fixed-size vector.
- **Non-Linear Solution**: Solves for global policy functions, capturing the interaction between inequality and the Zero Lower Bound (ZLB).
- **Neural Network Particle Filter**: A likelihood-based estimation method for non-linear heterogeneous agent models.

## Installation

```bash
git clone -b set_transformer https://github.com/EEbrami/estimating-hank-nn.git
cd estimating-hank-nn
pip install .
```

## Usage

### 1. Data Preparation

Download the required data (FRED Macro + SCF Micro) as described in `data/README.md`.

```bash
python data/process_data.py
```

### 2. Pre-training

Train the model on the deterministic steady state to ensure stability.

```bash
python examples/pretrain_hank.py
```

### 3. Full Training

Train the model over the full state space (Aggregate Shocks + Distribution).

```bash
python examples/train_hank.py
```

### 4. Estimation

Run the Neural Network Particle Filter to estimate parameters using real US data.

```bash
python examples/estimate_hank.py
```

## File Structure

- `src/estimating_hank_nn/`
  - `hank.py`: The `HANKModel` class (The "Brain").
  - `networks.py`: The `SetTransformer` architecture (The "Eye").
  - `particle_filter.py`: The estimation algorithm.
- `examples/`
  - `analytical.py`: The original RANK example (for reference).
  - `train_hank.py`: Main training loop for HANK.
  - `estimate_hank.py`: Estimation script.

## References

- Kase, H., Melosi, L., & Rottner, M. (2022). _Estimating Heterogeneous Agent Models with Neural Networks_.
- Kaplan, G., Moll, B., & Violante, G. L. (2018). _Monetary Policy According to HANK_.
- Zaheer, M., et al. (2017). _Deep Sets_.
