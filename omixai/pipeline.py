"""
OmiXAI — ensemble XAI pipeline for CNN and GNN models trained on genomic features.

Workflow:
    1. OmiXAI(model)                                   # model_type auto-detected
    2. .interpret(*loaders, method="hybrid"|"pfi")     # dims inferred from data
    3. .rank_features(feature_names)                   # skip inferred from names

Notes
-----
* No feature counts are passed in: the input width is read from the data, and
  the number of leading channels to skip (e.g. 4 one-hot DNA channels) is
  derived as ``F - len(feature_names)`` at ranking time.
* ``method="hybrid"`` runs the gradient/explainer ensemble + rank aggregation.
  ``method="pfi"`` runs permutation feature importance (GNN only).
* Parallelism is pure Python: hybrid interpretation overlaps CPU batch
  preparation via the DataLoader's ``num_workers``; PFI splits features across
  multiple CUDA devices when available (see ``omixai.xai.permutation_importance``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from captum.attr import Deconvolution, GuidedBackprop, InputXGradient, IntegratedGradients
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer
from .device import select_device

_DEVICE = select_device()

CNN_METHODS = ("IG", "IxG", "GB", "Deconv")
GNN_METHODS = ("IG", "IxG", "GB", "Deconv", "Saliency", "GNNExplainer")


def detect_model_type(model: nn.Module) -> str:
    """
    Infer whether a model is CNN or GNN by inspecting its layer types.

    Any module subclassing torch_geometric.nn.MessagePassing is treated as a GNN
    layer; nn.Conv2d presence indicates a CNN.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            return "gnn"
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return "cnn"
    raise ValueError(
        "Cannot detect model type automatically. "
        "Pass model_type='cnn' or model_type='gnn' explicitly."
    )


def _linear_edges(n: int) -> torch.Tensor:
    src, dst = [], []
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    return torch.tensor([src, dst], dtype=torch.long)


class OmiXAI:
    """
    Ensemble XAI pipeline for CNN and GNN models.

    Parameters
    ----------
    model      : trained PyTorch model (already trained — passed in as-is)
    model_type : 'cnn' or 'gnn'; auto-detected if omitted
    device     : defaults to CUDA if available
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or _DEVICE
        self.model  = model.to(self.device).eval()
        self.model_type = model_type or detect_model_type(model)
        if self.model_type not in ("cnn", "gnn"):
            raise ValueError("model_type must be 'cnn' or 'gnn'")
        self._scores: dict[str, np.ndarray] = {}   # method → full-width vector (len F)

    # ------------------------------------------------------------------ API

    def interpret(
        self,
        *loaders,
        method: str = "hybrid",
        methods: list[str] | None = None,
        width: int = 100,
        n_steps_ig: int = 50,
        pfi_n_repeats: int = 1,
        pfi_batch_size: int = 64,
        pfi_num_workers: int = 0,
        pfi_devices=None,
        gnn_explainer_epochs: int = 50,
    ) -> dict[str, np.ndarray]:
        """
        Compute feature scores. Stored vectors are full input width (length F);
        skipping of leading channels happens later in ``rank_features``.

        method="hybrid"
            Mean attribution over True-Positive predictions for each ensemble
            method. Pass several loaders to cover all TPs:
            ``interpret(train_loader, test_loader)`` or ``interpret(train_loader)``.
            DataLoader ``num_workers`` controls CPU-side parallelism.

        method="pfi"  (GNN only)
            Permutation feature importance: F1 drop when each feature is
            permuted. Parallelises across CUDA devices when more than one is
            visible (pure Python, no SLURM).
        """
        if method == "pfi":
            return self._interpret_pfi(
                *loaders, width=width, n_repeats=pfi_n_repeats,
                batch_size=pfi_batch_size, num_workers=pfi_num_workers,
                devices=pfi_devices,
            )
        if method != "hybrid":
            raise ValueError("method must be 'hybrid' or 'pfi'")

        available = CNN_METHODS if self.model_type == "cnn" else GNN_METHODS
        methods   = list(methods or available)
        unknown   = [m for m in methods if m not in available]
        if unknown:
            raise ValueError(
                f"Methods {unknown} not available for '{self.model_type}'. "
                f"Choose from: {available}"
            )

        for name in methods:
            self._gnn_explainer_epochs = gnn_explainer_epochs
            total, count, dim = None, 0, None
            for loader in loaders:
                t, c = (
                    self._run_cnn(loader, name, width, n_steps_ig)
                    if self.model_type == "cnn"
                    else self._run_gnn(loader, name, width)
                )
                if total is None:
                    total, dim = np.zeros_like(t), t.shape[0]
                total += t
                count += c
            self._scores[name] = total / max(count, 1)
            print(f"{name}: averaged over {count} TP regions ({dim} channels)")

        return self._scores

    def rank_features(
        self,
        feature_names: list[str] | None = None,
        scores: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """
        Hybrid ranking of stored (or provided) scores.

        Each method's scores are turned into percentage deviations from that
        method's mean, then averaged across methods; features are sorted
        descending (higher = more important).

        The number of leading channels to drop (DNA one-hot etc.) is inferred:
        ``skip = F - len(feature_names)``. If ``feature_names`` is None, all
        channels are ranked with positional names.
        """
        src = scores or self._scores
        if not src:
            raise RuntimeError("No scores — run interpret() first.")

        df = pd.DataFrame(src)                       # rows = full width F
        if feature_names is not None:
            skip = len(df) - len(feature_names)
            if skip < 0:
                raise ValueError(
                    f"feature_names ({len(feature_names)}) longer than scored "
                    f"channels ({len(df)})."
                )
            df = df.iloc[skip:].reset_index(drop=True)
            df.index = feature_names

        pct = pd.DataFrame(index=df.index)
        for col in df.columns:
            mu = df[col].mean()
            pct[col] = (df[col] - mu) / mu * 100 if mu != 0 else 0.0

        pct["mean_deviation"] = pct.mean(axis=1)
        return pct[["mean_deviation"]].sort_values("mean_deviation", ascending=False)

    # ------------------------------------------------------------------ PFI

    def _interpret_pfi(self, *loaders, width, n_repeats, batch_size,
                       num_workers, devices):
        if self.model_type != "gnn":
            raise ValueError("method='pfi' is implemented for GNN models only.")
        from .xai import permutation_importance

        data_list = []
        for loader in loaders:
            for dt in loader:
                x = dt.x.squeeze(0)                  # (N, F)
                y = dt.y.squeeze(0)                  # (N,)
                data_list.append(
                    Data(x=x, edge_index=_linear_edges(x.shape[0]), y=y)
                )
        # n_skip=0 → full-width vector (DNA channels included), kept consistent
        # with hybrid storage; rank_features slices the leading channels off.
        imp = permutation_importance(
            self.model, data_list, n_skip=0, n_repeats=n_repeats,
            batch_size=batch_size, num_workers=num_workers,
            devices=devices or [str(self.device)],
        )
        self._scores = {"PFI": imp}
        print(f"PFI: {len(imp)} channels over {len(data_list)} intervals")
        return self._scores

    # ------------------------------------------------------------------ CNN

    def _run_cnn(self, loader, method, width, n_steps_ig):
        explainer = self._build_captum(method)
        total, count = None, 0
        for x, y_true in tqdm(loader, desc=f"CNN/{method}", leave=False):
            x      = x.to(self.device)
            y_true = y_true.to(self.device).long()
            with torch.no_grad():
                pred = torch.argmax(self.model(x), dim=1).reshape(1, width)
            tp = [i for i in range(width) if pred[0][i] == y_true[0][i] == 1]
            if not tp:
                continue
            attrs = torch.squeeze(self._captum_attr(explainer, x, method, n_steps_ig), dim=0)
            tp_mean = attrs[tp, :].mean(dim=0)       # full width
            value = tp_mean.detach().cpu().numpy()
            total = np.zeros_like(value) if total is None else total
            total += value
            count += 1
        return (total if total is not None else np.zeros(1)), count

    # ------------------------------------------------------------------ GNN

    def _run_gnn(self, loader, method, width):
        explainer_obj = self._build_gnn_explainer(method)
        total, count = None, 0
        for dt in tqdm(loader, desc=f"GNN/{method}", leave=False):
            x    = dt.x.to(self.device)
            edge = dt.edge_index.to(self.device)
            y    = dt.y.to(self.device).long()
            valid = (edge < width).all(dim=0)
            edge  = edge[:, valid]
            with torch.no_grad():
                pred = torch.argmax(self.model(x, edge.squeeze()), dim=-1)
            tp = [i for i in range(width) if pred[0][i] == y[0][i] == 1]
            if not tp:
                continue
            mask = explainer_obj(x.squeeze(), edge).node_mask
            tp_mean = mask[tp, :].mean(dim=0)        # full width
            value = tp_mean.detach().cpu().numpy()
            total = np.zeros_like(value) if total is None else total
            total += value
            count += 1
        return (total if total is not None else np.zeros(1)), count

    # ------------------------------------------------------------------ builders

    def _build_captum(self, method):
        return {
            "IG":     IntegratedGradients,
            "IxG":    InputXGradient,
            "GB":     GuidedBackprop,
            "Deconv": Deconvolution,
        }[method](self.model)

    def _build_gnn_explainer(self, method) -> Explainer:
        _captum_names = {
            "IG":       "IntegratedGradients",
            "IxG":      "InputXGradient",
            "GB":       "GuidedBackprop",
            "Deconv":   "Deconvolution",
            "Saliency": "Saliency",
        }
        algo = (
            GNNExplainer(epochs=getattr(self, "_gnn_explainer_epochs", 50))
            if method == "GNNExplainer"
            else CaptumExplainer(_captum_names[method])
        )
        return Explainer(
            model=self.model,
            algorithm=algo,
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="log_probs",
            ),
        )

    @staticmethod
    def _captum_attr(explainer, x, method, n_steps_ig):
        if method == "IG":
            return explainer.attribute(x, target=1, n_steps=n_steps_ig)
        return explainer.attribute(x, target=1)
