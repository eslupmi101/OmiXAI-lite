"""
Thin runner for OmiXAI interpretation on a trained GraphMZC model.

One entry point, two formats via --method:
  --method hybrid   gradient/explainer ensemble + rank aggregation  (default)
  --method pfi      permutation feature importance (F1 drop per feature)

All heavy lifting lives in the package (omixai.OmiXAI); this script only loads
the genome, builds the canonical split, loads the model and saves results.

Data layout ($DATA_DIR): hg38_dna/, hg38_zdna/sparse/, hg38_features/sparse/.
Feature order is taken from OMIXAI_FEATURE_ORDER (the saved CSV) — required to
match the trained weights — and frozen to results/feature_names.json.

Usage:
  python scripts/run_interpret.py --model WEIGHTS --data_dir DIR \\
      --method hybrid --out_dir results/
  python scripts/run_interpret.py --model WEIGHTS --data_dir DIR \\
      --method pfi --pfi_intervals pos_train --out_dir results/pfi_dl/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root on path

from omixai import OmiXAI, GraphMZC, get_train_test_split_graph
from omixai.data import GraphGenomicDataset, load_genome_cached, CHROMS
from omixai.device import select_device


def main(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print(f"Device: {device}")

    DNA, ZDNA, DNA_features, feature_names = load_genome_cached(args.data_dir)
    n_features = len(feature_names)
    (out / "feature_names.json").write_text(json.dumps(feature_names))
    print(f"Omics features: {n_features} (order frozen → {out}/feature_names.json)")

    train_ds, test_ds = get_train_test_split_graph(
        args.width, CHROMS, feature_names, DNA, DNA_features, ZDNA
    )

    # Interpretation only makes sense on True Positives → positive train intervals.
    pos = [iv for iv in train_ds.intervals
           if ZDNA[iv[0]][int(iv[1]):int(iv[2])].any()]
    print(f"Train intervals: {len(train_ds.intervals)} total, {len(pos)} positive")
    interp_ds = GraphGenomicDataset(CHROMS, feature_names, DNA, DNA_features,
                                    ZDNA, pos, args.width)
    loader = DataLoader(interp_ds, batch_size=1, num_workers=args.num_workers,
                        shuffle=False)

    model = GraphMZC(n_features=n_features)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
    model = model.to(device).eval()
    print(f"Model loaded: {args.model}")

    xai = OmiXAI(model=model, model_type="gnn", device=device)

    if args.method == "pfi":
        xai.interpret(loader, method="pfi", width=args.width,
                      pfi_n_repeats=args.n_repeats, pfi_batch_size=args.batch_size,
                      pfi_num_workers=args.num_workers)
        scores = xai._scores["PFI"][4:]            # drop the 4 DNA channels
        np.save(out / "pfi_dl_scores.npy", scores)
        xai.rank_features(feature_names=feature_names).to_csv(out / "pfi_ranking.csv")
        print(f"Saved → {out}/pfi_dl_scores.npy, {out}/pfi_ranking.csv")
    else:
        # Run one method at a time, saving after each, so a timeout on the slow
        # GNNExplainer never loses the methods already computed.
        from omixai.pipeline import GNN_METHODS
        for m in GNN_METHODS:
            xai.interpret(loader, method="hybrid", methods=[m], width=args.width)
            np.save(out / "omixai_gnn_scores.npy", xai._scores)
            xai.rank_features(feature_names=feature_names).to_csv(out / "omixai_ranking.csv")
            print(f"[saved] methods so far: {list(xai._scores)}")
        print(f"\nTop 20:\n{xai.rank_features(feature_names).head(20).to_string()}")
        print(f"Done → {out}/omixai_ranking.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--method", choices=["hybrid", "pfi"], default="hybrid")
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--n_repeats", type=int, default=1, help="PFI permutation repeats")
    p.add_argument("--batch_size", type=int, default=64, help="PFI forward batch")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--out_dir", default="results/")
    main(p.parse_args())
