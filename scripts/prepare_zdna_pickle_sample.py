"""Prepare a bounded sample directly from the published z_dna pickle files.

This avoids expanding all 1,946 tracks.  It uses one chromosome and a small
set of real sparse omics tracks, while preserving the same 4 DNA channels,
100 bp windows and label semantics as the package datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data_sources/z_dna"))
    p.add_argument("--chrom", default="chr22")
    p.add_argument("--positive", type=int, default=100)
    p.add_argument("--negative", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("data/zdna_chr22_sample.npz"))
    args = p.parse_args()

    root = args.data_dir
    labels = joblib.load(root / "hg38_zdna/sparse/ZDNA_cousine.pkl")[args.chrom]
    dna_parts = sorted((root / "hg38_dna").glob(f"{args.chrom}_*.pkl"))
    if not dna_parts:
        raise FileNotFoundError(f"No DNA chunks for {args.chrom} in {root / 'hg38_dna'}")
    dna = "".join(joblib.load(path) for path in dna_parts)
    track_paths = sorted((root / "hg38_features/sparse").glob("*.pkl"))
    if not track_paths:
        raise FileNotFoundError("No omics pickle tracks downloaded")
    tracks = {path.stem: joblib.load(path)[args.chrom] for path in track_paths}
    names = list(tracks)
    width = 100
    rng = np.random.default_rng(args.seed)

    positive_starts = [int(i) for i, value in zip(labels.indices, labels.data) if value != 0]
    rng.shuffle(positive_starts)
    positive_starts = positive_starts[:args.positive]
    positive_starts = [min(max(0, start), len(dna) - width) for start in positive_starts]
    occupied = np.zeros(len(dna), dtype=bool)
    for start in positive_starts:
        occupied[start:start + width] = True
    negative_starts = []
    candidates = rng.permutation(np.arange(0, len(dna) - width, width))
    for start in candidates:
        start = int(start)
        if occupied[start:start + width].any() or labels[start:start + width].any():
            continue
        negative_starts.append(start)
        occupied[start:start + width] = True
        if len(negative_starts) == args.negative:
            break
    if len(positive_starts) < 2 or len(negative_starts) < 2:
        raise RuntimeError("Could not select enough positive and negative windows")
    windows = [(args.chrom, start, start + width, 1) for start in positive_starts]
    windows += [(args.chrom, start, start + width, 0) for start in negative_starts]
    rng.shuffle(windows)

    encoder = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))
    x_rows, y_rows = [], []
    for chrom, start, end, _ in windows:
        dna_ohe = encoder.transform(list(dna[start:end].upper()))
        omics = np.column_stack([tracks[name][start:end] for name in names]) / 1000.0
        x_rows.append(np.hstack((dna_ohe, omics)).astype(np.float32))
        y_rows.append(np.asarray(labels[start:end], dtype=np.int64))
    x = np.stack(x_rows)
    y = np.stack(y_rows)
    metadata = {"assembly": "hg38", "width": width, "seed": args.seed,
                "tracks": [{"name": name, "source": str(track_paths[i]),
                             "kind": "sparse_pickle", "scale": 1000}
                            for i, name in enumerate(names)],
                "windows": [{"chrom": c, "start": s, "end": e, "label": label}
                            for c, s, e, label in windows],
                "source": "vladislareon/z_dna", "chromosome": args.chrom}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, metadata=json.dumps(metadata))
    print(f"saved {args.output}: x={x.shape}, y={y.shape}, positives={sum(w[3] for w in windows)}")


if __name__ == "__main__":
    main()
