"""Small generated fixtures verify parsing; research runs require real data."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class LocalPreparationTest(unittest.TestCase):
    def test_fasta_bed_and_track_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fasta = base / "tiny.fa"
            fasta.write_text(">chr1\n" + "ACGT" * 50 + "\n")
            labels = base / "labels.bed"
            labels.write_text("chr1\t21\t24\nchr1\t41\t43\nchr1\t61\t64\n")
            peaks = base / "peaks.bed"
            peaks.write_text("chr1\t20\t25\n")
            manifest = base / "tracks.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["name", "path", "kind", "scale"])
                writer.writerow(["peak", "peaks.bed", "bed", 1])
            output = base / "sample.npz"
            subprocess.run([sys.executable, str(ROOT / "scripts/prepare_local_data.py"),
                            "--fasta", str(fasta), "--labels-bed", str(labels),
                            "--tracks-csv", str(manifest), "--positive", "3",
                            "--negative", "2", "--width", "10",
                            "--output", str(output)], check=True, capture_output=True)
            with np.load(output, allow_pickle=False) as sample:
                x, y = sample["x"], sample["y"]
                metadata = json.loads(str(sample["metadata"]))
            self.assertEqual(x.shape, (5, 10, 5))
            self.assertEqual(int(y.any(axis=1).sum()), 3)
            self.assertEqual(len(set((w["chrom"], w["start"])
                                     for w in metadata["windows"])), 5)
            for i, window in enumerate(metadata["windows"]):
                if window["start"] == 20:
                    np.testing.assert_array_equal(y[i], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
                    np.testing.assert_array_equal(x[i, :, 4], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    def test_stratified_split_is_disjoint(self):
        from scripts.run_local import split_indices
        y = np.zeros((20, 10), dtype=np.int64)
        y[:10, 0] = 1
        train, test = split_indices(y, 42)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(train) + len(test), 20)
        self.assertEqual(int(y[train].any(axis=1).sum()), 8)
        self.assertEqual(int(y[test].any(axis=1).sum()), 2)


if __name__ == "__main__":
    unittest.main()
