# Detailed Methodology Guide: Extending HANK with Neural Networks

This document expands on the outline by providing specific academic references and theoretical details for each section. Use this to write the "Methodology" section of your paper.

## 1. The Economic Environment (The HANK Model)

**Primary Reference:** Kaplan, G., Moll, B., & Violante, G. L. (2018). _Monetary Policy According to HANK_. American Economic Review.

You are implementing a standard One-Asset HANK model (simplified from KMV 2018). You must define the following:

### A. Households

Households maximize discounted utility subject to a budget constraint and a borrowing limit.

- **Objective**: $\max E_0 \sum_{t=0}^\infty \beta^t u(c_{it}, n_{it})$
- **Budget Constraint**: $c_{it} + b_{it+1} = w_t n_{it} + (1+r_{t-1})b_{it} + d_{it}$
- **Borrowing Constraint**: $b_{it+1} \ge \underline{b}$
- **Idiosyncratic Shock**: Labor productivity $e_{it}$ follows a Markov process (e.g., AR(1)).

**Key Concept to Mention**: The "Indirect Channel" (KMV 2018). In HANK, monetary policy works primarily by changing labor demand and thus labor income ($w_t n_{it}$), which strongly affects consumption for "hand-to-mouth" agents (those near $\underline{b}$).

### B. Firms

Standard New Keynesian firms subject to Rotemberg or Calvo pricing frictions.

- **Phillips Curve**: $\pi_t = \kappa X_t + \beta E_t \pi_{t+1}$
  - _Note_: In your nonlinear code, you use the nonlinear pricing condition, but the linearized form is often sufficient for exposition.

### C. Monetary Policy

- **Taylor Rule**: $R_t = \max(1, R^* (\frac{\pi_t}{\pi^*})^{\phi_\pi} (\frac{X_t}{X^*})^{\phi_y} e^{\epsilon_{m,t}})$
  - _Crucial_: We include a **Monetary Policy Shock** $\epsilon_{m,t}$ to capture interest rate volatility not explained by TFP.
  - _Crucial_: Highlight the **Zero Lower Bound (ZLB)** ($\max(1, \dots)$). This nonlinearity is why you need a Neural Network. Linearization methods (like Winberry) struggle here.

---

## 2. The Computational Challenge

**Reference:** Winberry, T. (2018). _A Method for Solving and Estimating Heterogeneous Agent Macroeconomic Models_. Quantitative Economics.

Explain _why_ this is hard.

- **The State Space**: The state is $S_t = (Z_t, \Gamma_t)$.
  - $Z_t$: Aggregate TFP (1 dimension).
  - $\Gamma_t$: The distribution of wealth/income (Infinite dimensions).
- **The Problem**: To solve the Bellman equation, agents need to forecast prices $(w, r)$, which depend on $\Gamma_{t+1}$.
- **Traditional Failure**: "Krusell-Smith" assumes $\Gamma_t$ is just the mean capital $K_t$. This fails when the _shape_ of the distribution matters (e.g., how many people are constrained at $\underline{b}$ during a recession).

---

## 3. The Neural Network Solution (The KMR Framework)

**Primary Reference:** Kase, H., Melosi, L., & Rottner, M. (2022). _Estimating Heterogeneous Agent Models with Neural Networks_.

**Secondary Reference (Architecture):** Lee, J., et al. (2019). _Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks_.

Describe your "Machine Learning" contribution here.

### A. The Set Transformer (Handling $\Gamma_t$)

How do you feed a histogram into a neural network without losing information about the "tails" (e.g., wealthy hand-to-mouth)? You use a **Set Transformer** with Attention.

**Theorem (Zaheer et al., 2017)**: A function $f(X)$ acting on a set $X$ is permutation-invariant if and only if it can be decomposed as $f(X) = \rho(\sum \phi(x))$.
**Improvement (Lee et al., 2019)**: While Zaheer uses simple summation, Lee uses **Multihead Attention** to capture higher-order interactions between elements.

- **Implementation**:
  1.  **Encoder (ISAB)**: We use **Induced Set Attention Blocks** (ISAB) to process the set of $N=1000$ agents. This reduces complexity from $O(N^2)$ to $O(N \cdot M)$ by using $M$ learnable "inducing points".
  2.  **Aggregator (PMA)**: We use **Pooling by Multihead Attention** (PMA) to aggregate the agent embeddings into a fixed-size vector. This learns a weighted combination rather than a simple mean.
  3.  **Decoder ($\rho$)**: A standard MLP that takes the summary vector and predicts aggregate variables.

### B. The "All-in-One" Expectation

Instead of the traditional "Nested Fixed Point" (guessing prices, solving Bellman, updating prices...), you train a single network to satisfy the equilibrium conditions directly.

- **Loss Function**: $\mathcal{L} = || \text{Euler Error} ||^2 + || \text{Market Clearing Error} ||^2$
- **Advantage**: It's much faster and allows for likelihood-based estimation.

---

## 4. Implementation Path: From RANK to HANK

**Clarification of Contribution**: The KMR repository provides a **RANK (Representative Agent)** example. This project extends it to **HANK**.

### A. The Foundation (Adapted from KMR)

We utilize the core infrastructure provided by KMR for the RANK model:

- **Training Loop**: The "All-in-One" minimization logic (Stochastic Gradient Descent on Euler residuals).
- **Network Skeleton**: The basic MLP structure for policy functions.
- **Particle Filter**: The sequential importance resampling algorithm.

### B. The Extension (Our Contribution)

To bridge the gap to HANK, we implemented:

1.  **The "Eye" (Set Transformer)**:
    - _Problem_: RANK only sees aggregate states ($\zeta$). HANK needs to see the distribution.
    - _Solution_: Implemented `SetTransformer` in `networks.py` to process $N=1000$ agents and extract a permutation-invariant embedding.
2.  **The "Brain" (HANKModel)**:
    - _Problem_: The `NKModel` class solves a 3-equation linear system.
    - _Solution_: Created `HANKModel` in `hank.py` which:
      - Takes the distribution embedding as input.
      - Solves the **Heterogeneous Agent Euler Equation** (approximated via aggregate consistency).
3.  **The "Logic" (Residuals)**:
    - _Problem_: RANK residuals are simple linear equations.
    - _Solution_: Implemented non-linear residuals including the **Zero Lower Bound** (Softplus approximation) and the **Portfolio Euler Equation**.

---

## 4. Estimation Strategy

**Reference:** Fernandez-Villaverde, J., & Rubio-Ramirez, J. F. (2007). _Estimating Macroeconomic Models: A Likelihood Approach_. Review of Economic Studies.

Explain how you take the model to data.

- **Particle Filter**: Since the model is nonlinear (ZLB) and non-Gaussian, we cannot use the Kalman Filter.
- **Process**:
  1.  Simulate $N$ particles (economies).
  2.  Compare model predictions (Output, Inflation) to real data.
  3.  Resample particles that match the data well.
  4.  Compute the likelihood $\mathcal{L}(\theta | \text{Data})$.

---

## Summary of Your Contribution

In your "Methodology" section, explicitly state:

> "We extend the HANK framework by replacing the linear approximation of the distribution (Winberry, 2018) with a non-linear **Set Transformer** (Lee et al., 2019). This allows us to capture the interaction between inequality and the Zero Lower Bound using attention mechanisms, without simplifying the heterogeneity."
