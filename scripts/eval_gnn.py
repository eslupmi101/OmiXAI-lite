"""Diagnostic eval: GLOBAL metrics (pooled over all nucleotides, comparable to
the paper) for several candidate feature orderings. A clear F1 winner tells us
which ordering matches the trained weights.

Feature order comes from genome_cache (OMIXAI_FEATURE_ORDER → saved CSV). The
extra "sorted" sweep is kept only as a sanity comparison."""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from torch_geometric.loader import DataLoader

from omixai import GraphMZC, get_train_test_split_graph
from genome_cache import load_genome_cached, CHROMS


def eval_global(model, loader, width, device):
    yt, yp, pr = [], [], []
    with torch.no_grad():
        for dt in loader:
            x = dt.x.to(device); edge = dt.edge_index.to(device)
            y = dt.y.to(device).long()
            valid = (edge < width).all(dim=0)
            out = model(x, edge[:, valid].squeeze())
            prob = torch.softmax(out, dim=-1)[..., 1]
            pred = out.argmax(dim=-1)
            yt += y.cpu().numpy().flatten().tolist()
            yp += pred.cpu().numpy().flatten().tolist()
            pr += prob.cpu().numpy().flatten().tolist()
    yt, yp, pr = np.array(yt), np.array(yp), np.array(pr)
    return dict(
        n=len(yt), pos=int(yt.sum()), pos_frac=float(yt.mean()),
        f1=f1_score(yt, yp, zero_division=0),
        auc=roc_auc_score(yt, pr) if yt.min() != yt.max() else float("nan"),
        prec=precision_score(yt, yp, zero_division=0),
        rec=recall_score(yt, yp, zero_division=0),
    )


def main(args):
    from omixai.device import select_device
    device = select_device(args.device)
    print(f"Device: {device}")

    np.random.seed(10)
    DNA, ZDNA, DNA_features, feat = load_genome_cached(args.data_dir)
    n_features = len(feat)
    print(f"Features: {n_features}")

    orders = {"as_loaded (feature order from genome_cache)": list(feat),
              "sorted": sorted(feat)}
    if args.feature_names and Path(args.feature_names).exists():
        fj = json.loads(Path(args.feature_names).read_text())
        if len(fj) == n_features:
            orders["feature_names.json"] = fj

    model = GraphMZC(n_features=n_features)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
    model = model.to(device).eval()
    print(f"Model loaded: {args.model}\n")

    for name, fn in orders.items():
        np.random.seed(10)  # identical split; only column order differs
        _, test = get_train_test_split_graph(args.width, CHROMS, fn, DNA, DNA_features, ZDNA)
        loader = DataLoader(test, batch_size=1, num_workers=4, shuffle=False)
        m = eval_global(model, loader, args.width, device)
        print(f"[{name}]  test nts={m['n']}, pos={m['pos']} ({m['pos_frac']:.1%})")
        print(f"    GLOBAL  F1={m['f1']:.4f}  AUC={m['auc']:.4f}  "
              f"P={m['prec']:.4f}  R={m['rec']:.4f}\n")

    print("Highest GLOBAL F1 = the ordering the weights were trained on "
          "(expect ~0.78 once the saved feature order is used).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--feature_names", default="results/feature_names.json")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    main(p.parse_args())
