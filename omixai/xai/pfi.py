"""
Permutation Feature Importance (PFI) for deep models on graph-structured
genomic intervals.

Importance of feature *i* = drop in F1 when column *i* is randomly permuted
across all nodes, relative to the unpermuted baseline. This is the deep-learning
analogue of classic PFI and is what Reviewer 1 asked us to compare against.

Speed:
  - BATCHED: intervals are stored as single-graph (N, F) Data and batched by PyG
    into one forward per batch (``batch_size`` graphs), not one per interval.
  - PARALLEL (pure Python, no SLURM): if more than one CUDA device is available
    (or ``devices`` is given), features are split across devices with
    ``torch.multiprocessing`` and computed concurrently. On a single GPU it runs
    in-process. ``num_workers`` parallelises CPU-side batch collation.

The function is dataset-agnostic: pass any list of PyG ``Data`` objects with
``x`` of shape (N, F), an ``edge_index`` and integer node labels ``y``.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from ..device import select_device


@torch.no_grad()
def _eval_f1(model, loader, device) -> float:
    yt, yp = [], []
    for b in loader:
        out = model(b.x.to(device), b.edge_index.to(device))
        yp.append(out.argmax(dim=-1).cpu().numpy())
        yt.append(b.y.numpy())
    return f1_score(np.concatenate(yt), np.concatenate(yp), zero_division=0)


def _pfi_on_device(
    model: torch.nn.Module,
    data_list: Sequence,
    feat_cols: Sequence[int],
    device: torch.device,
    n_repeats: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> np.ndarray:
    """Compute PFI for the given absolute feature columns on one device."""
    model = model.to(device).eval()
    loader = DataLoader(list(data_list), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    sizes = [d.x.shape[0] for d in data_list]
    base_f1 = _eval_f1(model, loader, device)

    g = torch.Generator().manual_seed(seed)
    imp = np.zeros(len(feat_cols), dtype=np.float64)
    for j, col in enumerate(feat_cols):
        orig = torch.cat([d.x[:, col] for d in data_list]).clone()
        drops = []
        for _ in range(n_repeats):
            perm = orig[torch.randperm(orig.numel(), generator=g)]
            off = 0
            for d, n in zip(data_list, sizes):
                d.x[:, col] = perm[off:off + n]; off += n
            drops.append(base_f1 - _eval_f1(model, loader, device))
        off = 0                                   # restore the original column
        for d, n in zip(data_list, sizes):
            d.x[:, col] = orig[off:off + n]; off += n
        imp[j] = float(np.mean(drops))
    return imp


def _worker(rank, devices, build_fn, state_dict, model_fn, splits,
            n_repeats, batch_size, num_workers, seed, ret_dir):
    import json
    from pathlib import Path
    device = torch.device(devices[rank])
    data_list = build_fn()
    model = model_fn()
    model.load_state_dict(state_dict)
    cols = splits[rank]
    imp = _pfi_on_device(model, data_list, cols, device,
                         n_repeats, batch_size, num_workers, seed + rank)
    np.save(Path(ret_dir) / f"pfi_rank_{rank}.npy", imp)
    json.dump({"cols": list(map(int, cols))},
              open(Path(ret_dir) / f"pfi_rank_{rank}.json", "w"))


def permutation_importance(
    model: torch.nn.Module,
    data_list: Sequence,
    *,
    n_skip: int = 4,
    n_repeats: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,
    devices: Sequence | None = None,
    seed: int = 0,
    build_fn: Callable[[], Sequence] | None = None,
    model_fn: Callable[[], torch.nn.Module] | None = None,
) -> np.ndarray:
    """
    Permutation importance over the non-skipped feature columns.

    Parameters
    ----------
    model       : trained model, called as ``model(x, edge_index)``.
    data_list   : list of PyG ``Data`` with x=(N, F), edge_index, y=(N,).
    n_skip      : leading columns to skip (e.g. 4 one-hot DNA channels). The
                  returned vector covers columns ``n_skip .. F-1``.
    n_repeats   : permutation repeats per feature (averaged).
    batch_size  : graphs per forward pass.
    num_workers : DataLoader workers for CPU-side batching.
    devices     : list of devices to parallelise features across (e.g.
                  ["cuda:0", "cuda:1"]). Defaults to all visible CUDA devices,
                  or a single device. Multi-device requires ``build_fn`` and
                  ``model_fn`` so each subprocess can build its own copy.
    build_fn    : zero-arg callable returning a fresh ``data_list`` (multi-GPU).
    model_fn    : zero-arg callable returning a fresh, untrained model (multi-GPU).

    Returns
    -------
    np.ndarray of shape (F - n_skip,) — importance per feature.
    """
    F = data_list[0].x.shape[1]
    feat_cols = list(range(n_skip, F))

    if devices is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = [str(select_device())]

    # single device → run in-process (the common cluster case: 1 GPU)
    if len(devices) == 1:
        return _pfi_on_device(model, data_list, feat_cols,
                              torch.device(devices[0]), n_repeats,
                              batch_size, num_workers, seed)

    # multi-device → split features across GPUs with torch.multiprocessing
    if build_fn is None or model_fn is None:
        raise ValueError(
            "Multi-device PFI needs build_fn and model_fn so each subprocess "
            "can construct its own data_list and model. Pass them, or set "
            "devices=['cuda:0'] to run single-device."
        )
    import tempfile
    import torch.multiprocessing as mp
    splits = [feat_cols[i::len(devices)] for i in range(len(devices))]
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as ret_dir:
        mp.spawn(
            _worker,
            args=(list(devices), build_fn, state_dict, model_fn, splits,
                  n_repeats, batch_size, num_workers, seed, ret_dir),
            nprocs=len(devices), join=True,
        )
        imp = np.zeros(len(feat_cols), dtype=np.float64)
        from pathlib import Path
        for rank in range(len(devices)):
            part = np.load(Path(ret_dir) / f"pfi_rank_{rank}.npy")
            cols = splits[rank]
            for local, col in enumerate(cols):
                imp[col - n_skip] = part[local]
    return imp
