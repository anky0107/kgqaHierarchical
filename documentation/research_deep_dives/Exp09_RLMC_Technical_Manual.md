# Experiment 9 Deep Dive: Adaptive Meta-Constraint Planning via Reinforcement Learning (RLMC)

## 1. Title
**RLMC: Reinforcement Learning Meta-Constraint Planning for Robust Multi-Hop Knowledge Graph Reasoning**

## 2. Abstract
State-of-the-art KGQA models often suffer from "Reasoning Drift" in multi-hop queries, where a single incorrect relation prediction leads to a permanent execution dead-end. Experiment 9 introduces the **RL Meta-Constraint Agent (RLMC)**, a policy-based Reinforcement Learning system that dynamically adjusts the search beam width ($K$) at each hop. By training a PPO (Proximal Policy Optimization) head atop a frozen RoBERTa backbone, the agent learns to "widen the net" when uncertain and "prune the search" when confident. This adaptive strategy achieved a landmark **76.29% Hits@1** on the CWQ dataset, significantly outperforming the static baseline of 65.03%.

---

## 3. The Objective: Efficiency vs. Accuracy
The fundamental trade-off in KG reasoning is the "Fan-out" problem.
*   **Too Broad ($K=100$)**: Guaranteed to find the answer, but consumes massive RAM and hits thousands of irrelevant entities.
*   **Too Narrow ($K=1$)**: Extremely fast, but one minor semantic error (choosing `born_in` instead of `place_of_birth`) causes a total failure.

**Experiment 9** replaces the fixed $K$ with an **Adaptive Agent** that looks at the question's state and selects one of 4 actions: `TIGHT`, `MEDIUM`, `LOOSE`, or `STOP`.

---

## 4. Architecture Specifications

### The Policy & Value Network
Implemented in [`train/exp9_rlmc.py`](file:///c:/Users/swoop/dev/res/kgqa/kgqaHierarchical/train/exp9_rlmc.py#L22):
*   **Shared Backbone**: Frozen RoBERTa-Large (1024-dim outputs).
*   **Policy Head**: A 2-layer MLP (`1024 -> 256 -> 4`) predicting categorical probabilities for search actions.
*   **Value Head**: A 2-layer MLP (`1024 -> 256 -> 1`) predicting the "Expected Reward" for the current state, used for PPO advantage calculation.

### Action Space Mapping
| Action | ID | Logical Width (K) | Use Case |
| :--- | :--: | :--- | :--- |
| **TIGHT** | 0 | **K = 1** | High confidence, direct fact retrieval. |
| **MEDIUM** | 1 | **K = 5** | Ambiguous terminology, multi-fact intersection. |
| **LOOSE** | 2 | **K ≈ 50** | Desperation mode, fallback to domain-wide search. |
| **STOP** | 3 | **K = 0** | Terminate reasoning path (prevent hallucinations). |

---

## 5. The Meta-Reward Mechanism (Line-by-Line Logic)

The agent is trained using a tiered reward signal computed in `calculate_meta_rewards()`:

1.  **Direct Jackpot (+1.0)**: If `Action=TIGHT` and `Top-1 == Gold`. The agent is rewarded for being efficient and perfect.
2.  **Safety Net (+0.5)**: If `Action=MEDIUM` and `Gold is inside Top-5`. The agent is rewarded for "catching" the answer but penalized for the memory overhead of the wider beam.
3.  **Domain Anchor (+0.1)**: If `Action=LOOSE` and `Gold is in Domain`. A very small reward to encourage survival over failure.
4.  **Terminal Reward (+1.0)**: If `Action=STOP` and the true path has ended.
5.  **Failure Spike (-1.0)**: If any action results in a dead-end or the correct relation is not captured in the chosen beam.

---

## 6. Training Pipeline (PPO Optimization)

### The Loss Function
$$Loss_{total} = Loss_{actor} + 0.5 \times Loss_{critic} + 0.01 \times Loss_{entropy}$$

*   **Actor Loss**: Uses Generalized Advantage Estimation (GAE) to push the weights towards actions that yielded higher-than-expected rewards.
*   **Critic Loss**: Mean Squared Error (MSE) to refine the agent's ability to predict its own success.
*   **Entropy Bonus**: Injects $0.01$ randomness to force the agent to explore `LOOSE` strategies occasionally rather than stagnating at `TIGHT`.

### Hyperparameters
*   **Epochs**: 50
*   **Learning Rate**: $1e-4$ (Aggressive, as the backbone is frozen).
*   **Gamma ($\gamma$)**: $0.99$ (Prioritizes long-term reasoning success).

---

## 7. Execution Benchmarks (Hits@1)

| Dataset | Mode | Hits@1 | Breakdown (1-hop / 2-hop / 3-hop) |
| :--- | :--- | :--- | :--- |
| **CWQ Test** | Execution | **76.29%** | 85.7% / 69.7% / 81.6% |

**Analysis**: The adaptive $K$ width is particularly effective on **3-hop questions** (81.6%), where static models typically collapse. The RLMC agent "survives" the complexity by opening the search beam at the critical bottleneck hop and then narrowing it down once the reasoning path stabilizes.

---

## 8. Real-World Execution Walkthrough (Inference)

When a question like *"Who directed the film that won Best Picture in 1994?"* enters the system:

1.  **Hop 1 (FILM)**: RLMC predicts `TIGHT` (K=1) for `award.award_category.nominees`.
2.  **Hop 2 (DIRECTOR)**: RLMC senses multiple potential director relations in the graph. It predicts `MEDIUM` (K=5) to ensure it captures `film.film.director` despite potential name clashes with `film.film.producer`.
3.  **Hop 3 (STOP)**: RLMC predicts `STOP` (Action 3). The BFS engine stops, and the final intersection of the graph walk returns the correct answer entity.
