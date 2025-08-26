## Project Context

Goal: Compare token-level vs sequence-level entropy maximization on a tiny LLM-like MDP via RL fine-tuning.

### Key ideas
- Entropy functional H(π) over sequences; first variation at π_k yields self-surprise reward: r(o_{1:T}|q) = -log π_k(o_{1:T}|q).
- Sequence-level objective with KL trust region per query: max_π E_q E_{o~π}[r] - β KL(π||π_k).
- AR-consistent practical updates either:
  - Closed-form unweighted (temperature): π_{k+1} ∝ π_k^{1 - 1/β} (token-level proxy).
  - Closed-form weighted (α-weights/entropy-to-go) to better approximate sequence entropy.
  - PPO approximations: simple PPO (token-level), episodic PPO (sequence-level).

### This repo (entropyfthf)
Implements GRPO using Hugging Face TRL on a toy Up/Down tree MDP:
- Tokenizer: custom 5-token vocab: <bos>, <eos>, <pad>, U, D.
- Env: UpDownTree with horizon H; first action U forces a line (only U), D keeps branching.
- Masking: `TreeLogitsProcessor` constrains generation to legal actions.
- Models: small GPT-2 policy; a reference model used for KL and reward.
- Reward: sequence-level self-surprise under current ref π_k: R = -log π_k(o_{1:T}|q); ref is resnapshotted every step.
- Trainer: TRL GRPOTrainer with per-step ref sync (trust region via β).

### How to run
- Create venv and install requirements (see README.md).
- Train (GRPO):
  ```bash
  python scripts/run_grpo_trl.py --steps 200 --horizon 8 --batch_size 64 --num_generations 4 --lr 1e-4 --beta 0.1
  ```
- Expected behavior: with sequence-level reward tied to π_k and resnapshotting, training increases sequence entropy while keeping a KL trust-region to the previous policy.

### Notes
- The reference model is frozen per run to stabilize KL and reward.
- You can resnapshot the ref between outer iterations if implementing iterative mirror descent.
- For comparisons, implement temperature or α-weighted closed-form updates in a sibling script.


