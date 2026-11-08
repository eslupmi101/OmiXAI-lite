"""
Thin runner for top-k feature-reduction retraining (paper Table 3).

Uses the package functions omixai.training.retrain_topk / retrain_topk_parallel
(GraphMZC, canonical stratified split, batch=32, seed=42, best-epoch).
Pick the feature ranking arm:

  OmiXAI arm:  --ranking results/omixai_ranking.csv
  PFI arm:     --pfi_scores results/pfi_dl/pfi_dl_scores.npy \\
               --feature_names results/feature_names.json

--k accepts one value or a comma list; use "all" for the full feature set:
  --k 100                 one k
  --k 50,100,300,500,700,all   sweep (default)

--parallel runs the k values concurrently, one process per CUDA device (needs a
multi-GPU allocation, e.g. --gres=gpu:v100:2). On a single GPU it runs
sequentially regardless.

Usage:
  python scripts/run_retrain.py --data_dir DIR --ranking results/omixai_ranking.csv
  python scripts/run_retrain.py --data_dir DIR --k 50,100,300,500,700,all --parallel
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root on path

from omixai.data import load_genome_cached, CHROMS
from omixai.training import retrain_topk, retrain_topk_parallel
from omixai.device import select_device


def _parse_k(text):
    if not text:
        return [50, 100, 300, 500, 700, None]
    out = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        out.append(None if tok in ("all", "none", "") else int(tok))
    return out


def main(args):
    ks = _parse_k(args.k)

    if args.pfi_scores:
        scores = np.load(args.pfi_scores)
        names = json.loads(Path(args.feature_names).read_text())
        ranked = [names[i] for i in np.argsort(scores)[::-1]]
        method = "PFI"
    else:
        import pandas as pd
        rank = pd.read_csv(args.ranking, index_col=0)["mean_deviation"]
        ranked = rank.sort_values(ascending=False).index.tolist()
        method = "OmiXAI"

    devices = (args.devices.split(",") if args.devices else
               [str(select_device(args.device))] if args.device != "auto" else None)
    if args.parallel:
        # workers reload the genome from cache; nothing huge is pickled.
        df = retrain_topk_parallel(
            args.data_dir, ranked, CHROMS, k=ks, devices=devices,
            width=args.width, n_epochs=args.n_epochs,
            num_workers=args.num_workers, method_name=method,
        )
    else:
        DNA, ZDNA, DNA_features, _ = load_genome_cached(args.data_dir)
        df = retrain_topk(ranked, CHROMS, DNA, DNA_features, ZDNA, k=ks,
                          width=args.width, n_epochs=args.n_epochs,
                          num_workers=args.num_workers, method_name=method,
                          device=select_device(args.device))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    single = f"_k{ks[0]}" if len(ks) == 1 and ks[0] is not None else ""
    suffix = single + ("_pfi" if args.pfi_scores else "_omixai")
    df.to_csv(out / f"retrain{suffix}.csv", index=False)
    print(f"\nSaved → {out}/retrain{suffix}.csv\n{df.to_string(index=False)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ranking", default="results/omixai_ranking.csv")
    p.add_argument("--pfi_scores", default=None)
    p.add_argument("--feature_names", default="results/feature_names.json")
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--k", default=None, help='e.g. "100" or "50,100,300,500,700,all"')
    p.add_argument("--parallel", action="store_true",
                   help="run k values concurrently, one process per CUDA device")
    p.add_argument("--devices", default=None,
                   help='comma list, e.g. "cuda:0,cuda:1"; default = all visible GPUs')
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--out_dir", default="results/")
    main(p.parse_args())
