# entropyfthf

Episodic PPO fine-tuning with TRL on a tiny Up/Down tree MDP, using a custom tokenizer and generation-time masking. Rewards are sequence-level self-surprise under a frozen reference model, and KL to the reference provides a trust-region.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_ppo_trl.py --steps 50 --horizon 8 --batch_size 64 --mini_batch_size 16 --lr 1e-4 --kl_coef 0.02
```

Notes:
- The tokenizer has tokens: <bos>, <eos>, <pad>, U, D.
- `TreeLogitsProcessor` enforces legal actions so that generated sequences respect the MDP.
- The reference model is frozen and used for KL and reward computation.

