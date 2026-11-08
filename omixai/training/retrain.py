"""
Top-k feature-reduction retraining (paper Table 3), as a package function.

Given a feature ranking (from OmiXAI hybrid or PFI), retrain the GNN from
scratch on the top-k omics features and report validation F1/AUC. This is the
"is a small feature subset enough?" experiment.

Model      : GraphMZC (the package GNN).
Split      : canonical class-stratified split (omixai.data.stratified_split_intervals).
Training   : batch=32, Adam(lr=1e-4, weight_decay=1e-4), NLLLoss, seed=42,
             report the best epoch (argmax val F1).

Dataset-agnostic: pass any dna/features/labels dicts keyed by chromosome (or any
sequence id) plus a feature ranking. ``k`` counts omics features; model input
dim is k + 4 DNA channels.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..models import GraphMZC
from ..device import select_device
from ..data import stratified_split_intervals


def set_random_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _linear_edge(width: int) -> torch.Tensor:
    src, dst = [], []
    for i in range(width - 1):
        src += [i, i + 1]; dst += [i + 1, i]
    return torch.tensor([src, dst], dtype=torch.long)


class _GDataset(torch.utils.data.Dataset):
    """Returns Data(x=(1, width, F), y=(1, width)); full-width intervals only."""

    def __init__(self, intervals, feats, dna, features, labels, width):
        self.iv = [iv for iv in intervals if int(iv[2]) - int(iv[1]) == width]
        self.feats, self.dna, self.features = feats, dna, features
        self.labels, self.w = labels, width
        self.le = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))

    def __len__(self):
        return len(self.iv)

    def __getitem__(self, i):
        c, b, e = self.iv[i][0], int(self.iv[i][1]), int(self.iv[i][2])
        ohe = self.le.transform(list(self.dna[c][b:e].upper()))
        cols = [self.features[f][c][b:e] for f in self.feats]
        X = (np.hstack((ohe, np.array(cols).T / 1000)).astype(np.float32)
             if cols else ohe.astype(np.float32))
        x = torch.tensor(X, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(self.labels[c][b:e], dtype=torch.int64).unsqueeze(0)
        return Data(x=x, y=y)


def _metrics(out, y):
    pred = out.argmax(dim=-1).cpu().numpy().flatten()
    prob = nn.Softmax(dim=-1)(out)[..., 1].detach().cpu().numpy().flatten()
    yt = y.cpu().numpy().flatten()
    if np.std(yt) == 0:
        return 0.0, f1_score(yt, pred, zero_division=0)
    return roc_auc_score(yt, prob), f1_score(yt, pred, zero_division=0)


def _run_epoch(model, opt, loader, edge, device, train):
    model.train() if train else model.eval()
    aucs, f1s = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for dt in loader:
            x, y = dt.x.to(device), dt.y.to(device).long()
            out = model(x, edge)
            a, f = _metrics(out, y)
            aucs.append(a); f1s.append(f)
            if train:
                opt.zero_grad()
                nn.NLLLoss()(out.permute(0, 2, 1), y).backward()
                opt.step()
    return float(np.mean(aucs)), float(np.mean(f1s))


def _train_and_eval(subset, dna, features, labels, train_iv, test_iv,
                    width, n_epochs, num_workers, device):
    set_random_seed(42)
    tr = DataLoader(_GDataset(train_iv, subset, dna, features, labels, width),
                    batch_size=32, num_workers=num_workers, shuffle=True)
    te = DataLoader(_GDataset(test_iv, subset, dna, features, labels, width),
                    batch_size=32, num_workers=num_workers, shuffle=False)
    edge = _linear_edge(width).to(device)

    set_random_seed(42)
    model = GraphMZC(n_features=len(subset)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best = {"f1": -1.0, "auc": 0.0, "epoch": 0}
    for ep in range(n_epochs):
        t0 = time.time()
        _run_epoch(model, opt, tr, edge, device, train=True)
        v_auc, v_f1 = _run_epoch(model, None, te, edge, device, train=False)
        if v_f1 > best["f1"]:
            best = {"f1": v_f1, "auc": v_auc, "epoch": ep + 1}
        print(f"  epoch {ep+1}/{n_epochs}  valF1={v_f1:.4f} valAUC={v_auc:.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return best


def _normalize_k(k):
    """Accept a single k (int or None) or an iterable of them; return a list."""
    if k is None or isinstance(k, int):
        return [k]
    return list(k)


def retrain_topk(
    ranked_features,
    chroms,
    dna,
    features,
    labels,
    *,
    k=(50, 100, 300, 500, 700, None),
    k_list=None,                       # deprecated alias for `k`
    width: int = 100,
    n_epochs: int = 10,
    neg_ratio: int = 2,
    num_workers: int = 4,
    seed: int = 42,
    method_name: str = "OmiXAI",
    device=None,
):
    """
    Retrain GraphMZC on top-k features for each requested k.

    Parameters
    ----------
    ranked_features : feature names ordered best-first (e.g.
                      ``ranking.index.tolist()`` from OmiXAI.rank_features).
    chroms          : sequence ids present in dna/features/labels.
    dna             : {seq_id -> str}
    features        : {feature_name -> {seq_id -> array}}
    labels          : {seq_id -> array of 0/1}
    k               : a single int (or None = "all"), or a list of them.

    Returns
    -------
    pandas.DataFrame with columns [k, method, best_epoch, f1, auc].
    """
    import pandas as pd

    ks = _normalize_k(k_list if k_list is not None else k)
    device = device or select_device()
    train_iv, test_iv = stratified_split_intervals(
        width, chroms, labels, neg_ratio=neg_ratio, test_size=0.2, seed=seed
    )
    print(f"[{device}] split: train={len(train_iv)} test={len(test_iv)}")

    rows = []
    for kv in ks:
        order = ranked_features if kv is None else ranked_features[:kv]
        subset = [f for f in order if f in features]
        kk = len(subset)
        print(f"\n=== {method_name} k={kv} ({kk} omics + 4 DNA) ===", flush=True)
        b = _train_and_eval(subset, dna, features, labels, train_iv, test_iv,
                            width, n_epochs, num_workers, device)
        rows.append({"k": kk, "method": method_name, "best_epoch": b["epoch"],
                     "f1": b["f1"], "auc": b["auc"]})
        print(f"  BEST: F1={b['f1']:.4f} AUC={b['auc']:.4f} @epoch {b['epoch']}")

    return pd.DataFrame(rows)


# --------------------------------------------------------------- parallel by k

def _retrain_worker(rank, devices, data_dir, ranked_features, chroms, k_splits,
                    width, n_epochs, neg_ratio, num_workers, seed, method_name, ret_dir):
    import torch
    from ..data import load_genome_cached
    dev = torch.device(devices[rank])
    dna, zdna, dna_features, _ = load_genome_cached(data_dir)
    df = retrain_topk(ranked_features, chroms, dna, dna_features, zdna,
                      k=k_splits[rank], width=width, n_epochs=n_epochs,
                      neg_ratio=neg_ratio, num_workers=num_workers, seed=seed,
                      method_name=method_name, device=dev)
    df.to_csv(Path(ret_dir) / f"retrain_rank_{rank}.csv", index=False)


def retrain_topk_parallel(
    data_dir,
    ranked_features,
    chroms,
    *,
    k=(50, 100, 300, 500, 700, None),
    devices=None,
    width: int = 100,
    n_epochs: int = 10,
    neg_ratio: int = 2,
    num_workers: int = 4,
    seed: int = 42,
    method_name: str = "OmiXAI",
):
    """
    Retrain across several k IN PARALLEL, one process per CUDA device.

    Pure Python (no SLURM): k values are round-robined across ``devices`` and
    each subprocess reloads the genome from the joblib cache, so nothing huge is
    pickled. Real speedup needs more than one visible GPU (request
    ``--gres=gpu:v100:N``); with a single device it transparently falls back to
    the sequential ``retrain_topk`` reload path.

    Returns a single concatenated DataFrame, sorted by k.
    """
    import pandas as pd

    if devices is None:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        devices = [f"cuda:{i}" for i in range(n)] if n > 1 else (
            ["cuda:0"] if n == 1 else [str(select_device())])

    ks = _normalize_k(k)

    # single device → no benefit from subprocesses; load once and run inline.
    if len(devices) == 1:
        from ..data import load_genome_cached
        dna, zdna, dna_features, _ = load_genome_cached(data_dir)
        return retrain_topk(ranked_features, chroms, dna, dna_features, zdna,
                            k=ks, width=width, n_epochs=n_epochs,
                            neg_ratio=neg_ratio, num_workers=num_workers,
                            seed=seed, method_name=method_name,
                            device=torch.device(devices[0]))

    import tempfile
    import torch.multiprocessing as mp
    splits = [ks[i::len(devices)] for i in range(len(devices))]
    splits = [s for s in splits if s]                      # drop empty shards
    devices = devices[:len(splits)]
    with tempfile.TemporaryDirectory() as ret:
        mp.spawn(
            _retrain_worker,
            args=(list(devices), data_dir, ranked_features, chroms, splits,
                  width, n_epochs, neg_ratio, num_workers, seed, method_name, ret),
            nprocs=len(devices), join=True,
        )
        dfs = [pd.read_csv(Path(ret) / f"retrain_rank_{r}.csv")
               for r in range(len(devices))]
    return pd.concat(dfs, ignore_index=True).sort_values("k").reset_index(drop=True)
