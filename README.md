# entropyfthf

Episodic GRPO fine-tuning with TRL on a tiny Up/Down tree MDP, using a custom tokenizer and generation-time masking. Rewards are sequence-level self-surprise (−log π_k(seq)) under the current reference, and KL to the reference provides a trust-region. The reference is resnapshotted every step (mirror-descent style).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_grpo_trl.py --steps 200 --horizon 8 --batch_size 64 --num_generations 4 --lr 1e-4 --beta 0.1 --eval_samples 1024
```

Notes:
- The tokenizer has tokens: <bos>, <eos>, <pad>, U, D.
- `TreeLogitsProcessor` enforces legal actions so that generated sequences respect the MDP.
- GRPO uses per-step resnapshotting of the reference and a reward tied to π_k.

## Plots

Exact sequence entropy over steps:
```bash
python scripts/plot_entropy.py --csv outputs/grpo_entropy_timeseries.csv --out outputs/grpo_entropy_over_steps.png --horizon 8
```

Sequence probability rank plots (before vs after):
```bash
python scripts/plot_sequence_distribution.py \
  --csv outputs/before_sequence_probs.csv outputs/after_sequence_probs.csv \
  --labels Before After \
  --align_by_first --logy \
  --out outputs/before_after_rankplot_logy.png
```


