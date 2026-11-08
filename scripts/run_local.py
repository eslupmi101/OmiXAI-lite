"""Train, interpret, rank and retrain GraphMZC on a compact real-data sample."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omixai import GraphMZC, OmiXAI
from omixai.device import select_device
from omixai.pipeline import GNN_METHODS


def split_indices(y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for positive in (False, True):
        ix = rng.permutation(np.flatnonzero(y.any(axis=1) == positive))
        if len(ix) < 2:
            raise ValueError("Need at least two positive and two negative windows")
        n_test = max(1, round(len(ix) * 0.2))
        test.extend(ix[:n_test].tolist())
        train.extend(ix[n_test:].tolist())
    return np.array(sorted(train)), np.array(sorted(test))


def edge_index(width: int):
    src = torch.arange(width - 1)
    return torch.stack((torch.cat((src, src + 1)), torch.cat((src + 1, src))))


def metrics(model, x, y, indices, edge, device, batch_size):
    model.eval()
    truths, preds, probabilities = [], [], []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(indices.tolist(), batch_size=batch_size):
            output = model(torch.from_numpy(x[batch]).to(device), edge)
            truths.append(y[batch].reshape(-1))
            preds.append(output.argmax(dim=-1).cpu().numpy().reshape(-1))
            probabilities.append(output.exp()[..., 1].cpu().numpy().reshape(-1))
    yt, yp, prob = map(np.concatenate, (truths, preds, probabilities))
    return {"f1": float(f1_score(yt, yp, zero_division=0)),
            "auc": float(roc_auc_score(yt, prob)) if len(np.unique(yt)) == 2 else None}


def train_model(x, y, train, test, device, *, epochs, batch_size, seed, checkpoint,
                positive_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GraphMZC(n_features=x.shape[-1] - 4).to(device)
    edge = edge_index(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    weight = torch.tensor([1.0, positive_weight], dtype=torch.float32, device=device)
    best = -1.0
    best_metrics = None
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = torch.utils.data.DataLoader(train.tolist(), batch_size=batch_size,
                                              shuffle=True, generator=generator)
        for batch in loader:
            xb = torch.from_numpy(x[batch]).to(device)
            yb = torch.from_numpy(y[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb, edge)
            loss = torch.nn.functional.nll_loss(output.reshape(-1, 2), yb.reshape(-1),
                                               weight=weight)
            loss.backward()
            optimizer.step()
        current = metrics(model, x, y, test, edge, device, batch_size)
        print(f"epoch {epoch}/{epochs}: val F1={current['f1']:.4f} "
              f"AUC={current['auc']}", flush=True)
        if current["f1"] > best:
            best = current["f1"]
            best_metrics = {"epoch": epoch, **current}
            if checkpoint:
                torch.save({key: tensor.detach().cpu() for key, tensor
                            in model.state_dict().items()}, checkpoint)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device)
    return model, best_metrics


def true_positive_windows(model, x, y, train, device, limit):
    model.eval()
    edge = edge_index(x.shape[1]).to(device)
    selected = []
    with torch.no_grad():
        for i in train:
            if not y[i].any():
                continue
            pred = model(torch.from_numpy(x[i:i + 1]).to(device), edge).argmax(-1)
            if ((pred[0].cpu().numpy() == 1) & (y[i] == 1)).any():
                selected.append(int(i))
                if len(selected) == limit:
                    break
    return selected


def probe_device(device, x):
    if device.type != "mps":
        return device
    try:
        model = GraphMZC(n_features=x.shape[-1] - 4).to(device)
        output = model(torch.from_numpy(x[:1]).to(device),
                       edge_index(x.shape[1]).to(device))
        output.sum().backward()
        del model, output
        torch.mps.empty_cache()
        return device
    except (RuntimeError, TypeError) as exc:
        print(f"MPS probe failed ({exc}); using CPU", flush=True)
        return torch.device("cpu")


def interpret(model, x, y, selected, device, names, args):
    edge = edge_index(x.shape[1])
    data = [Data(x=torch.from_numpy(x[i:i + 1]), y=torch.from_numpy(y[i:i + 1]),
                 edge_index=edge) for i in selected]
    loader = GraphLoader(data, batch_size=1, shuffle=False, num_workers=0)
    scores = {}
    devices = {}
    for method in GNN_METHODS:
        xai = OmiXAI(model, model_type="gnn", device=device)
        try:
            xai.interpret(loader, methods=[method], width=x.shape[1],
                          gnn_explainer_epochs=args.explainer_epochs)
        except (RuntimeError, TypeError) as exc:
            if device.type != "mps" or not any(word in str(exc).lower() for word in
                ("mps", "metal", "not implemented", "unsupported")):
                raise
            print(f"{method}: MPS unsupported ({exc}); retrying on CPU", flush=True)
            device = torch.device("cpu")
            xai = OmiXAI(model, model_type="gnn", device=device)
            xai.interpret(loader, methods=[method], width=x.shape[1],
                          gnn_explainer_epochs=args.explainer_epochs)
        scores[method] = xai._scores[method]
        devices[method] = str(device)
        np.savez_compressed(args.out_dir / "scores.npz", **scores)
        ranking = OmiXAI(model, model_type="gnn", device=device).rank_features(
            feature_names=names, scores=scores)
        ranking.to_csv(args.out_dir / "ranking.csv")
        (args.out_dir / "method_devices.json").write_text(json.dumps(devices, indent=2))
    return ranking, devices


def main():
    started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, default=Path("data/local_sample.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/local"))
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--retrain-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-interpret", type=int, default=10)
    p.add_argument("--explainer-epochs", type=int, default=10)
    p.add_argument("--top-k", default="10,25,50")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--positive-weight", type=float, default=25.0)
    args = p.parse_args()
    if min(args.epochs, args.retrain_epochs, args.batch_size,
           args.max_interpret, args.explainer_epochs, args.positive_weight) <= 0:
        p.error("epoch, batch and interpretation limits must be positive")
    if not args.sample.is_file():
        p.error(f"Missing real-data sample: {args.sample}; run prepare_local_data.py")
    with np.load(args.sample, allow_pickle=False) as sample:
        x, y = sample["x"], sample["y"]
        metadata = json.loads(str(sample["metadata"]))
    names = [track["name"] for track in metadata["tracks"]]
    if x.ndim != 3 or x.shape[:2] != y.shape or x.shape[2] != len(names) + 4:
        raise ValueError("Sample shape and track manifest disagree")
    if metadata["assembly"] != "hg38" or len(metadata["windows"]) != len(x):
        raise ValueError("Sample provenance is incomplete")
    top_ks = [int(value) for value in args.top_k.split(",")]
    if not top_ks or min(top_ks) < 1 or max(top_ks) > len(names):
        p.error(f"top-k must be between 1 and {len(names)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, test = split_indices(y, args.seed)
    device = probe_device(select_device(args.device), x)
    print(f"Device: {device}; sample: {len(x)} windows, {len(names)} tracks")
    checkpoint = args.out_dir / "checkpoint.pt"
    model, initial = train_model(x, y, train, test, device, epochs=args.epochs,
                                  batch_size=args.batch_size, seed=args.seed,
                                  checkpoint=checkpoint,
                                  positive_weight=args.positive_weight)
    selected = true_positive_windows(model, x, y, train, device, args.max_interpret)
    if not selected:
        raise RuntimeError("Model found no true-positive training windows; increase --epochs")
    print(f"Interpreting {len(selected)} true-positive training windows")
    ranking, method_devices = interpret(model, x, y, selected, device, names, args)
    rows = []
    for k in top_ks:
        feature_indices = [names.index(name) + 4 for name in ranking.index[:k]]
        subset_x = x[:, :, [0, 1, 2, 3] + feature_indices]
        _, result = train_model(subset_x, y, train, test, device,
                                epochs=args.retrain_epochs, batch_size=args.batch_size,
                                seed=args.seed, checkpoint=None,
                                positive_weight=args.positive_weight)
        rows.append({"k": k, **result})
    with (args.out_dir / "retrain.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "epoch", "f1", "auc"])
        writer.writeheader()
        writer.writerows(rows)
    run = {"sample": str(args.sample.resolve()), "train_indices": train.tolist(),
           "test_indices": test.tolist(), "interpreted_indices": selected,
           "device": str(device), "method_devices": method_devices,
           "initial_metrics": initial, "retrain_metrics": rows,
           "elapsed_seconds": round(time.perf_counter() - started, 1),
           "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                 (2**20 if platform.system() == "Darwin" else 1024), 1),
           "settings": {"epochs": args.epochs, "retrain_epochs": args.retrain_epochs,
                        "batch_size": args.batch_size, "top_k": top_ks,
                        "explainer_epochs": args.explainer_epochs,
                        "positive_weight": args.positive_weight}}
    (args.out_dir / "run.json").write_text(json.dumps(run, indent=2))
    print(f"Saved checkpoint, scores, ranking and retrain results to {args.out_dir}")


if __name__ == "__main__":
    main()
