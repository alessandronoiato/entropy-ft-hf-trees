import argparse
import csv
import glob
import json
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_timeseries_csv(path: str) -> List[Tuple[int, float]]:
    data: List[Tuple[int, float]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row.get("step", row.get("iteration", 0)))
            entropy = float(row.get("entropy_nats", row.get("entropy", 0.0)))
            data.append((step, entropy))
    data.sort(key=lambda x: x[0])
    return data


def read_json_glob(pattern: str) -> List[Tuple[int, float]]:
    items: List[Tuple[int, float]] = []
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        step = None
        for token in base.replace(".json", "").split("_"):
            if token.startswith("step"):
                try:
                    step = int(token.replace("step", ""))
                except ValueError:
                    pass
        with open(path, "r") as f:
            payload = json.load(f)
        entropy = float(payload.get("entropy_nats", payload.get("entropy", 0.0)))
        if step is None:
            # fallback: try reading explicit step from payload
            step = int(payload.get("step", 0))
        items.append((step, entropy))
    items.sort(key=lambda x: x[0])
    return items


def theoretical_max_entropy(horizon: int) -> float:
    if horizon < 2:
        return 0.0
    N = 2 + 3 * (2 ** (horizon - 2))
    return math.log(N)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="*", default=[], help="One or more CSV files with columns: step, entropy_nats")
    parser.add_argument("--json_glob", default=None, help="Glob to JSON files containing entropy_nats and step in filename or payload (deprecated in favor of CSV)")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels for each curve (match --csv order)")
    parser.add_argument("--out", default="outputs/entropy_over_steps.png", help="Output PNG path")
    parser.add_argument("--title", default="Exact sequence entropy over iterations")
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    series: List[List[Tuple[int, float]]] = []
    labels: List[str] = []

    for path in args.csv:
        series.append(read_timeseries_csv(path))
        labels.append(os.path.basename(path) if not args.labels else None)

    if args.json_glob:
        # Backward-compat: allow reading multiple JSON snapshots, but prefer CSV
        js = read_json_glob(args.json_glob)
        if js:
            series.append(js)
            labels.append(args.json_glob)

    # If custom labels provided
    if args.labels and len(args.labels) == len(series):
        labels = args.labels
    elif not labels or any(l is None for l in labels):
        labels = [f"run_{i+1}" for i in range(len(series))]

    plt.figure(figsize=(7, 4))
    for (steps, name) in zip(series, labels):
        if not steps:
            continue
        xs = [s for s, _ in steps]
        ys = [y for _, y in steps]
        plt.plot(xs, ys, label=name)

    # Theoretical maximum line
    hmax = theoretical_max_entropy(args.horizon)
    plt.axhline(hmax, color="gray", linestyle="--", linewidth=1, label=f"H_max (ln N), H={args.horizon}")

    plt.xlabel("Iteration (step)")
    plt.ylabel("Entropy (nats)")
    plt.title(args.title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


