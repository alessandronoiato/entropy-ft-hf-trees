import argparse
import csv
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_sequence_probs(path: str) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        # Expect columns: sequence, probability
        for row in reader:
            seq = str(row.get("sequence", ""))
            try:
                p = float(row.get("probability", 0.0))
            except Exception:
                p = 0.0
            items.append((seq, p))
    return items


def compute_metrics(seq_probs: List[Tuple[str, float]]):
    probs = [p for _, p in seq_probs]
    Z = sum(probs)
    if Z > 0:
        probs = [p / Z for p in probs]
    n = len(probs)
    H = -sum(p * math.log(p + 1e-40) for p in probs)
    H_max = math.log(n) if n > 0 else 0.0
    u = 1.0 / n if n > 0 else 0.0
    tv = 0.5 * sum(abs(p - u) for p in probs)
    return {"num_sequences": n, "entropy_nats": H, "entropy_max_nats": H_max, "tv_from_uniform": tv}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="One or more CSVs with columns: sequence, probability")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels for each CSV in order")
    parser.add_argument("--out", default="outputs/sequence_distribution.png", help="Output PNG path")
    parser.add_argument("--top_k", type=int, default=0, help="Use only top-K sequences per curve (0 = all)")
    parser.add_argument("--title", default="Sequence probability distribution (rank plot)")
    parser.add_argument("--logy", action="store_true", help="Use logarithmic y-scale")
    parser.add_argument(
        "--align_by_first",
        action="store_true",
        help="If set, use the first CSV's ordering to rank all curves (diagnostic view). Otherwise, rank each curve independently.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Read all curves first
    raw_curves: List[List[Tuple[str, float]]] = []
    for path in args.csv:
        seq_probs = read_sequence_probs(path)
        raw_curves.append(seq_probs)

    if not raw_curves:
        print("No data to plot.")
        return

    curves: List[Tuple[List[str], List[float]]] = []
    if args.align_by_first:
        # Determine a common rank order based on the first CSV (e.g., "before")
        if not raw_curves[0]:
            print("First CSV empty; cannot align by first.")
            return
        ref = list(raw_curves[0])
        ref.sort(key=lambda x: -x[1])
        if args.top_k and args.top_k > 0:
            ref = ref[: args.top_k]
        ref_order = [s for s, _ in ref]

        # Build aligned curves by mapping probabilities onto the ref order
        for seq_probs in raw_curves:
            d = {s: p for s, p in seq_probs}
            probs = [d.get(s, 0.0) for s in ref_order]
            Z = sum(probs) or 1.0
            probs = [p / Z for p in probs]
            if args.logy:
                eps = 1e-12
                probs = [max(p, eps) for p in probs]
            curves.append((ref_order, probs))
        x_label = "Rank (sorted by first CSV)"
    else:
        # Independent ranking per-curve
        for seq_probs in raw_curves:
            if not seq_probs:
                continue
            seq_probs = sorted(seq_probs, key=lambda x: -x[1])
            if args.top_k and args.top_k > 0:
                seq_probs = seq_probs[: args.top_k]
            labels = [s for s, _ in seq_probs]
            probs = [p for _, p in seq_probs]
            Z = sum(probs) or 1.0
            probs = [p / Z for p in probs]
            if args.logy:
                eps = 1e-12
                probs = [max(p, eps) for p in probs]
            curves.append((labels, probs))
        x_label = "Rank (sorted by own probability)"

    if not curves:
        print("No data to plot.")
        return

    # Labels handling
    if args.labels and len(args.labels) == len(curves):
        series_labels = args.labels
    else:
        series_labels = [os.path.basename(p) for p in args.csv[: len(curves)]]

    # Prepare rank plot (line) for readability
    max_n = max(len(probs) for _, probs in curves)
    plt.figure(figsize=(8, 4.5))
    for (labels, probs), name in zip(curves, series_labels):
        n = len(probs)
        xs = list(range(1, n + 1))
        plt.plot(xs, probs, marker="", linewidth=1.6, label=name)
    # Uniform line based on largest N (approx)
    uniform = 1.0 / max_n if max_n > 0 else 0.0
    plt.axhline(uniform, color="gray", linestyle="--", linewidth=1, label=f"Uniform 1/N (N≈{max_n})")
    plt.xlabel(x_label)
    plt.ylabel("Probability")
    plt.title(args.title)
    if args.logy:
        plt.yscale("log")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


