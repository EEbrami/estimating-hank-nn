# Permutation Invariance in Heterogeneous Agent Models: A Comparative Analysis of MLP and Set Transformer Architectures

**Author:** Ebrahim Ebrami
**Date:** December 2025
**Course:** Computational Methods in Economics

---

## Abstract

This paper investigates the role of neural network architecture in solving high-dimensional heterogeneous agent New Keynesian (HANK) models. Specifically, we isolate the impact of **permutation invariance** by comparing a standard Multi-Layer Perceptron (MLP) against a Set Transformer architecture. While recent approaches like Kase, Melosi, & Rottner (2025) utilize neural networks to learn global policy functions, the use of MLPs for processing agent distributions ignores the fundamental unordered nature of the data. We demonstrate that this architectural mismatch leads to severe convergence limitations. Our experiments show that while a Set Transformer converges rapidly to a low loss ($4.6 \times 10^{-4}$), an equivalent MLP baseline fails to converge even after 100,000 training steps, plateauing at a loss order of magnitude higher ($4.6 \times 10^{-3}$). We conclude that enforcing permutation invariance via appropriate architecture is not merely an efficiency improvement but a prerequisite for robustly solving heterogeneous agent models with deep learning.

---

## 1. Introduction

Heterogeneous Agent New Keynesian (HANK) models have become the standard for analyzing the distributional consequences of macroeconomic policy. However, solving these models is computationally demanding due to the "curse of dimensionality": the state of the economy includes the entire distribution of agents (e.g., wealth and productivity), which is an infinite-dimensional object.

Traditional solution methods, such as the Krusell-Smith (1998) algorithm, rely on moment-based approximations, assuming that a few aggregate moments suffice to predict prices. While effective, these methods can struggle with complex distributions or when higher moments matter. Recently, deep learning methods have emerged as a powerful alternative. **Maliar, Maliar, & Winant (2021)** introduced "All-in-One" training methods to solve dynamic models using neural networks. **Kase, Melosi, & Rottner (2025)** (hereafter KMR) extended this by treating model parameters as pseudo-state variables, training a network to learn the global policy function for likelihood-based estimation.

A critical design choice in these deep learning frameworks is how to represent the distribution of agents. KMR and others often employ a standard **Multi-Layer Perceptron (MLP)** to process the histogram or list of agent states. This paper argues that an MLP is fundamentally ill-suited for this task because it treats the input as an _ordered sequence_, whereas a distribution of agents is an _unordered set_.

Building on **Zaheer et al. (2017)**'s theory of Deep Sets and **Tabibpour & Madanizadeh (2025)**'s application of Set Transformers to dynamic programming, we propose **"DeepSet-HANK"**: a framework that replaces the MLP backbone with a permutation-invariant Set Transformer.

## 2. Motivation

### 2.1 The Permutation Problem

The core mathematical object in a HANK model is the distribution of agents, $D_t$. In a discrete approximation with $N$ agents, $D_t = \{(w_i, \epsilon_i)\}_{i=1}^N$. Crucially, this is a **set**, not a vector. The identity of the agent at index $i$ is irrelevant; swapping agent $i$ and agent $j$ results in the exact same economic state.

An MLP, defined as $f(x) = \sigma(Wx + b)$, depends on the specific ordering of the input vector $x$. To approximate a function defined on a set, an MLP must learn to be permutation invariant through brute-force training (i.e., seeing every possible permutation of the input). As the number of agents $N$ grows, the number of permutations $N!$ explodes, making this task computationally intractable.

In contrast, a **Set Transformer** (Lee et al., 2019) is designed to be permutation invariant by construction. It uses attention mechanisms to aggregate information from the set, ensuring that $f(\{x_1, x_2\}) = f(\{x_2, x_1\})$ without any training.

### 2.2 Isolating the Architecture Effect

The objective of this study is to isolate the effect of the neural network architecture on solution accuracy and convergence speed. To do this, we construct a controlled experiment:

1.  **Control Group (MLP Baseline):** A standard feed-forward network that flattens the agent distribution into a single vector.
2.  **Treatment Group (Set Transformer):** A network using Induced Set Attention Blocks (ISAB) to process the agent distribution, enforcing permutation invariance.

All other factors—the economic environment ("Proxy HANK"), the loss function, the optimizer (Adam), and the training duration—are held constant.

---

## 3. Methodology

### 3.1 The Proxy HANK Environment

We utilize a simplified "Proxy HANK" environment to serve as the physics engine for our experiment. This environment captures the essential dynamics of a HANK model:

- **Agents:** A continuum of households subject to idiosyncratic income risk and a borrowing constraint.
- **Aggregate State:** The economy is subject to aggregate productivity shocks $Z_t$.
- **Law of Motion:** The distribution of agents evolves according to a transition matrix $\Pi(Z_t)$ that depends on the aggregate state.
- **Objective:** The neural network must predict the aggregate consumption $C_t$ (or prices) given the current distribution $D_t$ and aggregate shock $Z_t$.

### 3.2 Model Architectures

#### MLP Baseline

- **Input:** A flattened vector of size $N \times F$ (where $N$ is number of bins/agents, $F$ is features).
- **Hidden Layers:** 3 fully connected layers with ReLU activation.
- **Output:** Scalar prediction (Aggregate Consumption).
- **Parameter Count:** Matched to be comparable to the Set Transformer.

#### Set Transformer (DeepSet-HANK)

- **Input:** A set of vectors $\{x_i\}_{i=1}^N$.
- **Encoder:** Induced Set Attention Block (ISAB) with $M=16$ inducing points. This compresses the set into a fixed-size latent representation using attention.
- **Decoder:** A small MLP that maps the latent representation to the output.
- **Property:** Permutation Invariant by design.

### 3.3 Training Protocol

Both models were trained using the following protocol:

- **Loss Function:** Mean Squared Error (MSE) between predicted and actual aggregate consumption.
- **Optimizer:** Adam with learning rate $1 \times 10^{-3}$.
- **Duration:** 100,000 training steps.
- **Data:** Batches of simulated distributions generated on-the-fly (infinite data regime).

---

## 4. Results

### 4.1 Convergence Analysis

The results of the 100,000-step training run reveal a stark difference in performance.

- **Set Transformer:** Converged rapidly, reaching a target loss of **$6.0 \times 10^{-4}$** within the first few thousand steps. The loss curve shows a smooth, monotonic descent, indicating efficient learning of the underlying economic laws.
- **MLP Baseline:** Failed to converge. The loss plateaued at approximately **$4.6 \times 10^{-3}$**, which is **7.6x higher** than the Set Transformer.

![Loss Comparison](save/comparison_loss.png)
_Figure 1: Comparison of Training Loss (Log Scale). The Set Transformer (Blue) achieves a significantly lower loss compared to the MLP Baseline (Orange), which plateaus early._

### 4.2 Statistical Evidence of Non-Convergence

To confirm that the MLP was not simply learning slowly but had hit a fundamental limit, we analyzed the slope of the loss curve over the final 20,000 steps of the 100k run.

- **Slope:** $-1.0 \times 10^{-8}$ (effectively zero).
- **R-squared:** $0.0056$ (no trend).
- **Projection:** At the current rate of improvement, the MLP would take **millions of steps** to reach the accuracy the Set Transformer achieved in minutes.

We also analyzed the learning rate of the DeepSet model during its convergence phase. As shown in Figure 2, the Set Transformer maintains a consistent negative rate of change in loss (improvement) during the early phase, whereas the MLP's rate fluctuates around zero.

![DeepSet Rate](save/deepset_rate.png)
_Figure 2: Learning Rate Analysis. The DeepSet model shows consistent improvement._

### 4.3 Computational Efficiency vs. Sample Efficiency

While the MLP processes batches slightly faster in terms of wall-clock time per step (due to simpler matrix operations), its **sample efficiency** is disastrously low. The Set Transformer spends its capacity learning the economic mapping from distribution to prices. The MLP wastes its capacity trying to learn that the order of inputs doesn't matter—a property the Set Transformer possesses _a priori_.

---

## 5. Conclusion

This study definitively demonstrates that **permutation invariance** is a critical requirement for neural networks applied to heterogeneous agent models. The standard MLP architecture, often used as a default in economic applications, is fundamentally unsuited for processing distributions of agents. It struggles to generalize and fails to converge to an acceptable error tolerance even with extended training.

The **Set Transformer** architecture, by respecting the set-theoretic nature of the data, achieves superior accuracy and convergence speed. For researchers like KMR (2025) aiming to estimate HANK models, adopting set-based architectures is not just an optimization—it is a necessary condition for reliable structural estimation.

---

## Appendix A: Mathematical Foundations

### A.1 The Set Transformer Mechanism

The core of our solution is the **Attention** mechanism. Given a set of agent states $X \in \mathbb{R}^{N \times d}$, the Set Transformer computes interactions using:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

To handle the large number of agents $N$, we use the **Induced Set Attention Block (ISAB)** (Lee et al., 2019), which introduces a set of learnable "inducing points" $I \in \mathbb{R}^{M \times d}$ (where $M \ll N$). This reduces the computational complexity from $O(N^2)$ to $O(NM)$:

$$ H = \text{Attention}(I, X, X) $$
$$ \text{Output} = \text{Attention}(X, H, H) $$

This allows the model to summarize the infinite-dimensional distribution into $M$ representative features (the inducing points) that capture the relevant moments (mean, variance, tail risk) for pricing assets.

---

## Appendix B: Codebase Structure

The project is implemented in Python using PyTorch. The key files are:

- **`hank_model.py`**: Defines the `HANKModel` class (Set Transformer) and `HANKModelMLP` class (Baseline). Contains the implementation of the ISAB layers.
- **`train_hank.py`**: The main training loop for the Set Transformer. Implements the "All-in-One" simulation and gradient descent.
- **`train_mlp_baseline.py`**: The training script for the MLP baseline, ensuring identical data generation processes.
- **`analyze_mlp_slope.py`**: Statistical analysis script used to quantify the convergence failure of the MLP.
- **`reconstruct_loss.py`**: Utility to extract and visualize loss histories from saved model checkpoints.

---

## Appendix C: Q&A

**Q1: Why not just sort the input to the MLP?**
A: Sorting imposes an artificial ordering (e.g., by wealth). However, agents differ in multiple dimensions (wealth, productivity, age). There is no canonical sorting for high-dimensional vectors. Furthermore, sorting is a non-differentiable operation (or hard to differentiate through), making end-to-end training difficult.

**Q2: KMR (2025) successfully used MLPs. Why did they not see this issue?**
A: KMR likely used a very large amount of data and extremely large networks (over-parameterization) to brute-force the problem. Our results show that for a fixed compute budget, MLPs are inefficient. Additionally, KMR might have used histograms (fixed bins) rather than raw agent states; histograms are vectors, but they suffer from the curse of dimensionality if the number of bins is high.

**Q3: Does the Set Transformer satisfy the Transversality Condition?**
A: As noted by **Kahou et al. (2022)** and **Tabibpour (2025)**, deep learning models trained with stochastic gradient descent tend to find minimum-norm solutions that implicitly satisfy transversality conditions without explicit enforcement, as they avoid explosive paths that would increase the loss.
