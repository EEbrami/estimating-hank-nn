# Executive Summary: The "DeepSet-HANK" Project

**To:** Professor [Name]
**From:** Ebrahim Ebrami
**Subject:** Final Project Narrative – Improving KMR (2025) with Set Transformers

---

## 1. The Context (My Presentation)

In my presentation (**"Estimating Nonlinear Heterogeneous Agent Models with Neural Networks"**), I introduced the methodology of **Kase, Melosi, & Rottner (2025)**.

- **The Goal:** To estimate HANK models using likelihood-based methods (Bayesian estimation).
- **The Bottleneck:** Solving the model for every parameter draw is too slow.
- **The KMR Solution:** They treat parameters $\theta$ as "Pseudo-State Variables." They train a Neural Network to learn the global policy function $\pi(S_t, \theta)$ _once_, allowing for instant evaluation during estimation.

## 2. The Critique (The "Permutation Problem")

While replicating the KMR framework, I identified a theoretical weakness in their neural network architecture. They use a standard **Multi-Layer Perceptron (MLP)** to process the distribution of agents.

- **The Problem:** An MLP treats the input as an _ordered list_. However, a distribution of agents is an _unordered set_.
- **The Evidence:** As **Zaheer et al. (2017)** state in _Deep Sets_:
  > "A typical machine learning algorithm... is designed for fixed dimensional data instances. Their extensions to handle [sets]... must be permutation invariant."
- **The Flaw:** An MLP is **not** permutation invariant. It must "waste" training resources learning that swapping Agent A and Agent B doesn't matter.

## 3. The Innovation ("DeepSet-HANK")

For my final project, I replaced the MLP backbone with a **Set Transformer**, following **Tabibpour et al. (2025)**.

- **Why it works:** **Tabibpour (2025)** explicitly notes that while previous methods (like DeepHAM) work, they:
  > "...do not address permutation invariance as effectively."
- **The Solution:** I implemented the **Induced Set Attention Block (ISAB)** from **Lee et al. (2019)**. This architecture allows the network to "attend" to clusters of agents (e.g., the borrowing constrained) regardless of their order in the input vector.
- **Quote:** Tabibpour (2025) confirms:
  > "The Set Transformer demonstrated strong performance... making it particularly well-suited for complex economic models where the relationships between agents... involve intricate dependencies."

## 4. The Implementation

I built a custom codebase from scratch (using PyTorch) that integrates:

1.  **The KMR Estimator:** For the "All-in-One" training loop (citing _Maliar et al., 2021_).
2.  **The Set Transformer:** For the "Brain" of the agent.
3.  **A "Proxy HANK" Environment:** A simplified physics engine (citing _Kaplan et al., 2018_) to benchmark the architecture's performance.

## 5. Conclusion

This project is not just a replication; it is a **methodological refinement**. By applying a state-of-the-art architecture (Set Transformer) to the KMR framework, I demonstrate a more robust way to solve the "Curse of Dimensionality" in heterogeneous agent models, directly addressing the core theme of our **Computational Methods** course.
