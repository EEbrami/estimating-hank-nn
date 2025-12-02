# DeepSet-HANK: Methodology & Implementation Guide

This document serves as the technical backbone for the **DeepSet-HANK** project. It details the theoretical foundations, the specific "Proxy HANK" benchmark environment, and the neural network architecture used to estimate the model.

---

## 1. Introduction & Motivation

**The Problem:** Heterogeneous Agent New Keynesian (HANK) models are computationally expensive to estimate because the state space includes an infinite-dimensional distribution of agents $\Gamma_t$. Traditional methods (Krusell-Smith) approximate this distribution with its first moment (mean capital), which fails to capture the rich distributional dynamics that drive HANK results (e.g., the "wealthy hand-to-mouth").

**The Existing Solution (KMR 2025):** Kase, Melosi, & Rottner propose a "Neural Network Particle Filter" that treats structural parameters as pseudo-state variables. However, their implementation relies on standard Multi-Layer Perceptrons (MLPs) to process the distribution.

**The Limitation:** MLPs are **permutation-sensitive**. They treat the input vector $[x_1, x_2, ...]$ as an ordered sequence. However, a distribution of agents is inherently an **unordered set**. Swapping Agent A and Agent B in the input should not change the aggregate state of the economy. MLPs must "learn" this invariance, which is inefficient.

**Our Innovation ("DeepSet-HANK"):** We replace the KMR MLP backbone with a **Set Transformer** (Tabibpour 2025, Lee 2019). This architecture is theoretically guaranteed to be permutation-invariant, making it the mathematically correct tool for processing agent distributions.

**The Experiment:** We benchmark this new architecture on a **"Proxy HANK"** environment—a simplified physics engine that isolates the computational performance of the estimator under wide parameter uncertainty and monetary shocks.

---

## 2. Theoretical Foundations (The 4 Pillars)

To build the "DeepSet-HANK" estimator, we synthesize four distinct strands of literature:

### Pillar 1: The Estimator (KMR 2025)

- **Source:** Kase, H., Melosi, L., & Rottner, M. (2025). _Estimating Nonlinear Heterogeneous Agent Models with Neural Networks_.
- **Concept:** **Pseudo-State Variables**. Instead of solving the model for fixed parameters $\theta$, we solve for a global policy function $\pi(s, \theta)$ where $\theta$ is treated as a state variable.
- **Math:**
  $$ C_t = \pi(S_t, \theta | W) $$
    Where $S_t$ is the economic state and $W$ are the neural network weights.
- **Benefit:** "Solve once, estimate many times." This allows us to estimate the model using a Particle Filter without re-solving the equilibrium at every step.

### Pillar 2: The Architecture (Set Transformers)

- **Source:** Tabibpour, A., et al. (2025). _Solving High-Dimensional Dynamic Programming Using Set Transformer_.
- **Source:** Lee, J., et al. (2019). _Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks_.
- **Concept:** **Permutation Invariance**. A function $f(X)$ acting on a set $X$ is permutation-invariant if $f(\pi(X)) = f(X)$ for any permutation $\pi$.
- **Implementation:** We use the **Induced Set Attention Block (ISAB)**.
  $$ H = \text{MAB}(I, X) $$
    $$ O = \text{MAB}(X, H) $$
    Where $X$ is the set of $N$ agents, $I$ are $M$ learnable "inducing points" (prototypes), and MAB is Multihead Attention.
- **Why it fits HANK:** It allows the network to "attend" to specific parts of the distribution (e.g., the borrowing constrained agents) regardless of where they appear in the input vector.

### Pillar 3: The Physics (Proxy HANK)

- **Source:** Kaplan, G., Moll, B., & Violante, G. L. (2018). _Monetary Policy According to HANK_.
- **Concept:** **The "Proxy HANK" Benchmark**.
  - _Full HANK:_ Requires finding the interest rate $R_t$ such that the bond market clears: $B_t^d(R_t) = B_t^s$.
  - _Proxy HANK (Our Approach):_ We train the network to satisfy the **Aggregate Euler Equation** of a Representative Agent, but **conditioned on the heterogeneous distribution**.
  - **Dynamics:** To ensure the distribution $\Gamma_t$ evolves endogenously (giving the Set Transformer a dynamic signal to learn), we implement a **Reiter-style proxy law of motion**:
    $$ \Gamma*{t+1} = \rho \Gamma_t + \alpha Z_t + \epsilon*{idio} $$
    This ensures that aggregate productivity shocks ($Z_t$) drive changes in inequality, mimicking the "wealth effect" of business cycles.
- **Justification:** This simplification creates a stable, controlled environment to test if the Set Transformer can effectively learn the mapping from **Distributions $\to$ Aggregate Prices**, without the numerical instability of the full market clearing loop. It isolates the _architectural_ performance.

### Pillar 4: The Training (All-in-One)

- **Source:** Maliar, L., Maliar, S., & Winant, P. (2021). _Deep Learning for Solving Dynamic Economic Models_.
- **Concept:** **All-in-One Expectation Operator**.
- **Problem:** Evaluating expectations $\mathbb{E}_t[\cdot]$ usually requires expensive quadrature integration ($N^2$ operations).
- **Solution:** Replace the integral with **Stochastic Gradient Descent (SGD)**. We minimize the residual of two random draws of future shocks $\epsilon', \epsilon''$:
  $$ \min \mathcal{L} = || f(s, \epsilon') \cdot f(s, \epsilon'') ||^2 $$
- **Benefit:** Reduces computational cost to $O(1)$ per training step, enabling us to train on massive datasets of simulated economies.

---

## 3. Implementation Details

### A. Network Architecture (`networks.py`)

The `HANKNet` class implements the Set Transformer architecture:

1.  **Input:** A 3D Tensor `(Batch, Agents, Features)` where Features = $(b_{it}, a_{it}, e_{it})$.
2.  **Encoder (ISAB):** Two layers of Induced Set Attention Blocks with 32 inducing points. This compresses the information from $N=1000$ agents into 32 "prototype" agents.
3.  **Aggregator (PMA):** A Pooling by Multihead Attention layer that aggregates the prototypes into a single latent vector $Z$.
4.  **Policy Head (MLP):** A standard MLP that takes $[Z, \text{Aggregate State}, \text{Parameters}]$ and predicts $[Output Gap, Inflation]$.

### B. Training Loop (`hank.py`)

The training loop minimizes the weighted sum of squared residuals:

1.  **NKPC Residual:** $\pi_t - (\kappa X_t + \beta \mathbb{E}_t \pi_{t+1})$
2.  **Euler Residual:** $X_t - (\mathbb{E}_t X_{t+1} - \frac{1}{\sigma}(R_t - \mathbb{E}_t \pi_{t+1} - r_t^n))$
3.  **ZLB Constraint:** We use a **Softplus Approximation** for the Zero Lower Bound to maintain differentiability:
    $$ R_t = 1 + \frac{1}{\kappa} \log(1 + e^{\kappa(R^\* - 1)}) $$

### C. Safety Mechanisms

To ensure robust training for long runs (20,000+ iterations):

- **Checkpointing:** Model weights are saved every 1000 steps.
- **CSV Logging:** Loss history is written to disk incrementally to prevent data loss.
- **Wide Parameter Ranges:** We train on a "Wide Net" of parameters (e.g., $\sigma \in [0.5, 5.0]$) to ensure the estimator is robust to regime changes.

### D. Computational Complexity Analysis

A critical consideration in computational economics is the cost of the solution method. We analyze the trade-off between our Set Transformer approach and the standard MLP baseline.

1.  **Per-Step Complexity:**

    - **Standard Attention:** $O(N^2)$. Computing attention between all $N$ agents is prohibitively expensive for large $N$ (e.g., $N=1000$).
    - **MLP (KMR Baseline):** $O(N)$. Standard methods typically compute moments (mean, variance) which takes linear time, or feed a flattened vector.
    - **Set Transformer (Ours):** $O(N \cdot M)$. By using **Induced Set Attention Blocks (ISAB)** with $M=32$ inducing points, we reduce the complexity to be linear in the number of agents.
    - _Result:_ Our method is computationally tractable, adding only a constant factor overhead compared to simple moment-based methods, while retaining the full distributional information.

2.  **Sample Efficiency (The "Free Lunch"):**
    - **Theoretical Basis:** **Zaheer et al. (2017)** establish that for a function defined on a set to be valid, it _must_ be permutation invariant.
    - **The Trade-off:**
      - **MLP:** Relies on the _Universal Approximation Theorem_. Given infinite data and infinite width, it can approximate any function, including a permutation-invariant one. However, in practice (finite data), it struggles to generalize because it overfits to the specific ordering of the training examples (Tabibpour et al., 2025).
      - **Set Transformer:** Has the correct **Inductive Bias** hard-coded into the architecture. It restricts the hypothesis space to _only_ permutation-invariant functions.
    - **Conclusion:** This reduces the "sample complexity" of the problem. The Set Transformer requires fewer gradient steps to reach a given error threshold because it does not need to "learn" the symmetry of the problem from scratch.

---

## 4. Preliminary Analysis (Dry Run Results)

A dry run of 200 iterations confirmed the stability of the "DeepSet-HANK" architecture:

- **Convergence:** The loss function exhibits a monotonic downward trend, indicating the Set Transformer is successfully extracting relevant features from the distribution.
- **Robustness:** The model successfully handles the **Monetary Policy Shock** ($m\_shock$), a feature absent in the original KMR code but critical for HANK analysis.
- **Efficiency:** The "All-in-One" operator allows the model to process batches of 64 economies with 1000 agents each in milliseconds, enabling a full training run in ~2.5 hours on a standard CPU.

---

## 5. Conclusion

**DeepSet-HANK** represents a methodological advance in the estimation of heterogeneous agent models. By integrating **Set Transformers** (Architecture) with the **KMR Framework** (Estimator) and testing on a **Proxy HANK** (Physics) benchmark, we demonstrate a scalable path forward for solving high-dimensional economic models.
