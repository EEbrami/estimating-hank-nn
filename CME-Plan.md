# CME Project Plan: Extending HANK with Neural Networks

## Progress Log (What has been done)

### 1. Data Preparation (Completed)

- **Macro Data**: Downloaded Real GDP, PCE Inflation, and Federal Funds Rate from FRED.
- **Micro Data**: Downloaded SCF 2022 Summary Extract (`rscfp2022.dta`).
- **Processing**: Created `process_data.py` to load both datasets and compute initial distributional moments (e.g., Liquid Wealth Share).

### 2. Model Implementation (Completed)

- **Architecture**: Implemented `HANKModel` class in `src/estimating_hank_nn/hank.py`.
- **Heterogeneity**: Added `SetTransformer` in `src/estimating_hank_nn/networks.py` to process agent distributions $(b, a, e)$ in a permutation-invariant way.
- **Estimation**: Implemented `ParticleFilter` in `src/estimating_hank_nn/particle_filter.py` for likelihood evaluation.

### 3. Verification (Completed)

- **Pre-training**: Successfully ran `examples/pretrain_hank.py` to verify the model can learn the steady state.
- **Pipeline**: Successfully ran `examples/estimate_hank.py` to verify the estimation code runs end-to-end (even if results are preliminary).

### 4. Advanced Verification (Completed)

- **Checkpointing**: Implemented and verified `save_every` logic to prevent data loss during long runs (`dry_run_log_v2.md`).
- **Robustness**: Verified model stability under "Wide Net" parameter ranges ($\sigma \in [0.5, 5.0]$) and **Monetary Policy Shocks** (`dry_run_log_v3.md`).
- **Sync**: Ensured estimation and training scripts are fully synchronized (`dry_run_log_v4.md`).

---

## What is Ahead (The "CME-Plan")

### 1. Full Model Training (Critical Next Step)

- **Current Status**: The model is only pre-trained on the steady state. This is why the likelihoods are `-inf`—the model doesn't yet know how to react to large shocks.
- **Action**: Run `examples/train_hank.py` for a significant number of iterations (e.g., 10,000+) to learn the global policy function.

### 2. Methodology Refinement

- **Current Status**: The `residuals` function currently uses a simplified aggregation.
- **Action**: Explicitly define the "Investment Wedge" or the specific HANK equilibrium conditions.
  - _Question_: Do we want to add a loss term that specifically targets the _evolution_ of the wealth distribution? Currently, the `SetTransformer` takes the distribution as input, but we primarily minimize aggregate errors.

### 3. Final Estimation & Analysis

- **Action**: Once the model is trained, re-run `examples/estimate_hank.py` to get valid likelihoods and estimate $\sigma$.
- **Action**: Generate the final plots using `examples/analyze_hank.py`.

---

## 4. The Research Strategy: A Novel Synthesis

This section details the theoretical contribution of the project, combining three distinct elements.

### The "Marriage" Proposal

We are combining the following elements to create a novel contribution:

- **The Framework (The "Groom"): Kase, Melosi, & Rottner (2022)**.

  - **Contribution**: They provide the **methodology** for estimating HANK models (Neural Network Particle Filter, Likelihood function, Training loop).
  - **Current Weakness**: They typically use simpler architectures (like moments or basic Deep Sets) to handle the distribution.

- **The Architecture (The "Bride"): Tabibpour (2025) / Lee et al. (2019)**.

  - **Contribution**: Tabibpour proves that the **Set Transformer** (Lee et al.) is superior to Deep Sets for solving heterogeneous agent models.
  - **Why**: It captures higher-order interactions between agents (e.g., how the rich affect the poor via prices) using _Attention_ mechanisms, whereas Deep Sets just sums them up.

- **The "Physics" (The House): Kaplan, Moll, & Violante (2018)**.
  - **Contribution**: This provides the **HANK model** equations (budget constraints, Phillips curve) that define the rules of the simulation.

### Correction: "Zahir" vs. Lee

It is important to distinguish between the architectures:

1.  **Zaheer et al. (2017): Deep Sets**. This is the simpler architecture that sums up agent states ($\sum \phi(x)$). This is the baseline we are improving upon.
2.  **Lee et al. (2019): Set Transformer**. This is the advanced architecture that uses _Multihead Attention_ ($Attention(Q, K, V)$) to process sets. This is what we are using.

### The Value Add

Merging these papers solves a specific problem in HANK estimation:

- **The Problem**: In HANK, the _shape_ of the distribution matters. Simple moments or Deep Sets might compress the distribution too much, losing information about "hand-to-mouth" agents who drive the transmission mechanism.
- **The Solution**: The Set Transformer uses **Attention**. It can learn to "attend" specifically to the agents at the borrowing constraint ($b=0$) because they matter most for aggregate consumption.
- **The Result**: The KMR-style estimator becomes more accurate because its "Eye" (the Set Transformer) sees the relevant economic heterogeneity better than the standard "Eye" (Deep Sets).

### Implementation Warning

- **Computational Expense**: The KMR algorithm runs a particle filter (thousands of model evaluations).
- **Complexity**: Set Transformer is $O(N^2)$.
- **The Fix**: We use the **Induced Set Attention Block (ISAB)** (as described in Lee 2019) to reduce complexity to $O(N \cdot M)$ by using a small set of "inducing points" to summarize the distribution.

---

## Recommendations / Q&A

### "Do I need more data?"

**No.** You have the standard set for this type of analysis:

- **Macro**: GDP, Inflation, Rates (captures business cycles).
- **Micro**: SCF (captures the wealth distribution shape).
  Unless your specific research question involves a specific asset class (e.g., housing vs. stocks), this is sufficient.

### "Do I need to layout the methodology?"

**Yes, absolutely.**
The current code implementation makes some implicit assumptions (e.g., that the `SetTransformer` sufficiently captures the state). You should write down:

1.  **The Agent's Problem**: What are they maximizing? (Standard consumption-savings).
2.  **The Aggregation**: How does the neural network approximate the aggregate laws of motion?
3.  **The Loss Function**: Explicitly state that you are minimizing the residuals of the Euler equation and NKPC, conditioned on the distributional embedding.

### "What should I do now?"

1.  **Start Training**: Run `python examples/train_hank.py`. This will take time.
2.  **Write Methodology**: While the model trains, draft the "Methodology" section of your paper, focusing on how the Neural Network replaces the traditional "Winberry" or "Reiter" method for handling distributions.
