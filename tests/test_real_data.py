"""Tests for the M4b real-dataset (.ts) parser and loader (R9 wiring).

Uses a small hand-written ``.ts`` fixture rather than the network-downloaded
UEA archive, so these tests need no internet access. The fixture's format was
verified directly against the real downloaded ``BasicMotions_TRAIN.ts``
(6 colon-blocks of 100 comma-values + a trailing string label) -- this test
uses a smaller 2-channel, 3-timestep version of the same shape.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from causaltemp_xai.real_data import load_real_dataset, parse_ts_file

_FIXTURE_TS = """\
#comment line, must be ignored
@problemName Toy
@timeStamps false
@missing false
@univariate false
@dimensions 2
@equalLength true
@seriesLength 3
@classLabel true A B
@data
1.0,2.0,3.0:4.0,5.0,6.0:A
-1.0,-2.0,-3.0:-4.0,-5.0,-6.0:B
"""


class TestParseTsFile:
    def test_parses_shape_and_values(self, tmp_path):
        p = tmp_path / "toy.ts"
        p.write_text(_FIXTURE_TS, encoding="utf-8")
        X, labels = parse_ts_file(p)
        assert X.shape == (2, 3, 2)  # (N, T, k)
        assert labels.tolist() == ["A", "B"]
        # instance 0, channel 0 (first colon-block) -> [1.0, 2.0, 3.0]
        assert np.allclose(X[0, :, 0], [1.0, 2.0, 3.0])
        # instance 0, channel 1 (second colon-block) -> [4.0, 5.0, 6.0]
        assert np.allclose(X[0, :, 1], [4.0, 5.0, 6.0])
        assert np.allclose(X[1, :, 0], [-1.0, -2.0, -3.0])

    def test_raises_on_missing_data_marker(self, tmp_path):
        p = tmp_path / "bad.ts"
        p.write_text("@problemName Toy\nno data marker here\n", encoding="utf-8")
        with pytest.raises(ValueError):
            parse_ts_file(p)


class TestLoadRealDataset:
    def test_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_real_dataset("nonexistent", out_dir=tmp_path)

    def test_roundtrip_omits_graph_and_mechanism(self, tmp_path):
        dest = tmp_path / "toy"
        dest.mkdir()
        rng = np.random.default_rng(0)
        for split, n in (("train", 6), ("val", 2), ("test", 2)):
            np.save(dest / f"X_{split}.npy", rng.normal(size=(n, 3, 2)))
            np.save(dest / f"Y_{split}.npy", rng.integers(0, 2, size=n))
        with open(dest / "meta.json", "w") as fh:
            json.dump({"class_names": ["A", "B"]}, fh)

        out = load_real_dataset("toy", out_dir=tmp_path)
        assert out["graph"] is None
        assert out["mechanism"] is None
        assert out["X_train"].shape == (6, 3, 2)
        assert out["Y_val"].shape == (2,)
        assert out["meta"]["class_names"] == ["A", "B"]
