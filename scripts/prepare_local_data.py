"""Build a bounded, real-data OmiXAI sample from hg38 FASTA and BED tracks.

Coordinates are zero-based, half-open. Inputs are never loaded in full: FASTA
uses its .fai index and each feature track is scanned once. No synthetic values
are created; missing signal in a track is represented by zero.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_fai(path: Path) -> dict[str, tuple[int, int, int, int]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            chrom, length, offset, bases, bytes_per_line, *_ = line.rstrip("\n").split("\t")
            result[chrom] = tuple(map(int, (length, offset, bases, bytes_per_line)))
    if not result:
        raise ValueError(f"Empty FASTA index: {path}")
    return result


def build_fai(fasta: Path, index_path: Path) -> None:
    """Create a minimal faidx index without loading the genome into memory."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta.open("rb") as source, index_path.open("w") as destination:
        chrom, length, offset, bases, bytes_per_line = None, 0, 0, 0, 0
        while True:
            position = source.tell()
            line = source.readline()
            if not line:
                break
            if line.startswith(b">"):
                if chrom is not None:
                    destination.write(f"{chrom}\t{length}\t{offset}\t{bases}\t{bytes_per_line}\n")
                chrom = line[1:].split()[0].decode("ascii")
                length, offset, bases, bytes_per_line = 0, source.tell(), 0, 0
            else:
                sequence = line.rstrip(b"\r\n")
                if not bases:
                    bases, bytes_per_line = len(sequence), len(line)
                length += len(sequence)
        if chrom is not None:
            destination.write(f"{chrom}\t{length}\t{offset}\t{bases}\t{bytes_per_line}\n")


def read_sequence(handle, index, chrom: str, start: int, end: int) -> str:
    length, offset, bases, bytes_per_line = index[chrom]
    if start < 0 or end > length:
        raise ValueError(f"Interval outside {chrom}: {start}-{end}")
    result = bytearray()
    pos = start
    while pos < end:
        line_pos = pos % bases
        count = min(end - pos, bases - line_pos)
        handle.seek(offset + (pos // bases) * bytes_per_line + line_pos)
        result.extend(handle.read(count))
        pos += count
    return result.decode("ascii").upper()


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: expected at least 3 tab columns")
            try:
                yield fields[0], int(fields[1]), int(fields[2]), fields
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid coordinates") from exc


def read_tracks(path: Path):
    tracks = []
    with path.open(newline="") as handle:
        for entry in csv.DictReader(handle):
            name = entry["name"].strip()
            source = Path(entry["path"].strip()).expanduser()
            if not source.is_absolute():
                source = path.parent / source
            kind = entry["kind"].strip().lower()
            if kind not in ("bed", "bedgraph"):
                raise ValueError(f"{name}: kind must be bed or bedgraph")
            if not source.is_file():
                raise FileNotFoundError(source)
            scale = float(entry.get("scale") or 1)
            if not name or scale <= 0:
                raise ValueError(f"Invalid track name or scale: {entry}")
            tracks.append((name, source.resolve(), kind, scale))
    if not tracks or len({t[0] for t in tracks}) != len(tracks):
        raise ValueError("Feature manifest must contain unique, nonempty names")
    return tracks


def choose_windows(labels_bed, index, width, n_positive, n_negative, seed):
    positive = set()
    label_intervals = defaultdict(list)
    for chrom, start, end, _ in rows(labels_bed):
        if chrom not in index or start < 0 or end <= start or end > index[chrom][0]:
            continue
        label_intervals[chrom].append((start, end))
        for window_start in range(start // width * width, end, width):
            if window_start + width <= index[chrom][0]:
                positive.add((chrom, window_start))
    if len(positive) < n_positive:
        raise ValueError(f"Only {len(positive)} valid positive windows; requested {n_positive}")

    rng = np.random.default_rng(seed)
    selected_pos = [sorted(positive)[i] for i in rng.choice(len(positive), n_positive, replace=False)]
    chromosomes = sorted({chrom for chrom, _ in selected_pos})
    selected_neg = set()
    attempts = 0
    while len(selected_neg) < n_negative and attempts < n_negative * 100:
        chrom = chromosomes[int(rng.integers(len(chromosomes)))]
        window = int(rng.integers(index[chrom][0] // width)) * width
        candidate = chrom, window
        if candidate not in positive:
            selected_neg.add(candidate)
        attempts += 1
    if len(selected_neg) < n_negative:
        raise ValueError("Could not sample enough non-Z-DNA windows")
    windows = sorted(set(selected_pos) | selected_neg)
    for intervals in label_intervals.values():
        intervals.sort()
    return windows, label_intervals


def add_intervals_to_array(path, kind, scale, windows, width, target):
    by_chrom = defaultdict(list)
    for i, (chrom, start) in enumerate(windows):
        by_chrom[chrom].append((start, i))
    starts = {chrom: [start for start, _ in pairs] for chrom, pairs in by_chrom.items()}
    for chrom, start, end, fields in rows(path):
        if chrom not in by_chrom or end <= start:
            continue
        value = 1.0 if kind == "bed" else float(fields[3]) / scale
        if not np.isfinite(value):
            raise ValueError(f"Non-finite feature value in {path}")
        pairs = by_chrom[chrom]
        left = max(0, bisect_left(starts[chrom], start - width + 1))
        for win_start, i in pairs[left:]:
            if win_start >= end:
                break
            a, b = max(start, win_start) - win_start, min(end, win_start + width) - win_start
            if a < b:
                target[i, a:b] = value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", type=Path, required=True, help="uncompressed hg38 FASTA with .fai index")
    p.add_argument("--fai", type=Path, default=None, help="optional existing FASTA index")
    p.add_argument("--labels-bed", type=Path, required=True, help="hg38 Kouzine Z-DNA BED")
    p.add_argument("--tracks-csv", type=Path, required=True, help="name,path,kind,scale")
    p.add_argument("--output", type=Path, default=Path("data/local_sample.npz"))
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--positive", type=int, default=200)
    p.add_argument("--negative", type=int, default=400)
    p.add_argument("--max-features", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.width, args.positive, args.negative, args.max_features) < 1:
        p.error("width, positive, negative and max-features must be positive")
    if not args.fasta.is_file() or not args.labels_bed.is_file():
        p.error("FASTA and labels BED must exist")
    fai = args.fai or Path(f"{args.fasta}.fai")
    if not fai.is_file():
        fai = args.output.parent / "reference.fai"
        print(f"Indexing FASTA once to {fai}", flush=True)
        build_fai(args.fasta, fai)
    index = read_fai(fai)
    tracks = read_tracks(args.tracks_csv)[:args.max_features]
    windows, label_intervals = choose_windows(
        args.labels_bed, index, args.width, args.positive, args.negative, args.seed)
    n, width = len(windows), args.width
    x = np.zeros((n, width, 4 + len(tracks)), dtype=np.float32)
    y = np.zeros((n, width), dtype=np.int64)
    bases = "ACGT"
    with args.fasta.open("rb") as handle:
        for i, (chrom, start) in enumerate(windows):
            sequence = read_sequence(handle, index, chrom, start, start + width)
            for pos, base in enumerate(sequence):
                if base in bases:
                    x[i, pos, bases.index(base)] = 1
            for a, b in label_intervals[chrom]:
                if a >= start + width:
                    break
                if b > start:
                    y[i, max(0, a - start):min(width, b - start)] = 1
    if int((y.any(axis=1)).sum()) != args.positive:
        raise RuntimeError("Positive-window count changed during label construction")
    nonzero_counts = {}
    for j, (name, source, kind, scale) in enumerate(tracks, 4):
        print(f"Feature {j - 3}/{len(tracks)}: {name}", flush=True)
        add_intervals_to_array(source, kind, scale, windows, width, x[:, :, j])
        nonzero_counts[name] = int(np.count_nonzero(x[:, :, j]))
    if not any(nonzero_counts.values()):
        raise ValueError("All feature tracks are zero on selected windows; check hg38 coordinates")
    metadata = {
        "assembly": "hg38", "width": width, "seed": args.seed,
        "fasta": str(args.fasta.resolve()), "labels_bed": str(args.labels_bed.resolve()),
        "tracks": [{"name": name, "path": str(source), "kind": kind, "scale": scale}
                   for name, source, kind, scale in tracks],
        "nonzero_counts": nonzero_counts,
        "windows": [{"chrom": chrom, "start": start, "end": start + width}
                    for chrom, start in windows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, metadata=json.dumps(metadata))
    print(f"Saved {args.output}: {n} real windows, {len(tracks)} features, "
          f"{x.nbytes / 2**20:.1f} MiB uncompressed")


if __name__ == "__main__":
    main()
