"""Data-contract tests for Phase 08's graph-quality figure (``fig6``).

Added 2026-08-12 with the M4e re-derivation, which switched both of fig6's
axes from the raw ``graph_error``/``propagation_error`` columns to their
``*_sigma`` companions. That swap has one dangerous failure mode: if the
expected columns are missing the function returns ``False`` and prints a skip
line, which in a long pipeline log looks exactly like "nothing to plot yet".
These tests pin both branches -- it plots when the scale-free columns are
there, and it skips *loudly and returns False* when they are not, rather than
falling back to the scale-confounded columns the re-derivation retired.
"""

from __future__ import annotations

import csv
import importlib

_phase08 = importlib.import_module("experiments.08_aggregate_and_report")

_LADDER = [0.0, 0.25, 0.5, 0.75, 1.0]

_SIGMA_COLS = {
    "graph_error_sigma": 1.0,
    "graph_error_sigma_lo": 0.9,
    "graph_error_sigma_hi": 1.1,
    "propagation_error_sigma_min": 0.05,
    "propagation_error_sigma_max": 0.5,
}


def _write_table(tables_dir, config="full_nl", with_sigma=True):
    path = tables_dir / f"table_graph_quality_{config}.csv"
    rows = []
    for i, frac in enumerate(_LADDER):
        row = {
            "config": config,
            "label": f"corrupt_frac={frac:g}",
            "corrupt_frac": frac,
            "n_seeds": 3,
            "shd_mean": float(i * 5),
            "graph_error": 0.001 * i,
            "graph_error_lo": 0.0009 * i,
            "graph_error_hi": 0.0011 * i,
            "frac_vacuous": 0.0,
            "frac_vacuous_lo": 0.0,
            "frac_vacuous_hi": 0.0,
            "propagation_error_min": 0.0,
            "propagation_error_max": 0.18,
        }
        if with_sigma:
            row.update({k: v * (i or 1) if "graph" in k else v for k, v in _SIGMA_COLS.items()})
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


class TestFig6DataContract:
    def test_plots_when_scale_free_columns_present(self, tmp_path):
        tables = tmp_path / "tables"
        tables.mkdir()
        _write_table(tables)
        out = tmp_path / "fig6.pdf"

        assert _phase08.fig6_graph_quality_curve(tables, out) is True
        assert out.exists() and out.stat().st_size > 0

    def test_skips_when_only_the_retired_columns_are_present(self, tmp_path, capsys):
        """A pre-2026-08-12 table must NOT be plotted with the raw columns as
        a silent fallback -- the scale confound is the whole reason fig6's
        axes changed. Skip, say why, write nothing."""
        tables = tmp_path / "tables"
        tables.mkdir()
        _write_table(tables, with_sigma=False)
        out = tmp_path / "fig6.pdf"

        assert _phase08.fig6_graph_quality_curve(tables, out) is False
        assert not out.exists()
        assert "_sigma" in capsys.readouterr().out

    def test_skips_when_no_table_exists(self, tmp_path):
        tables = tmp_path / "tables"
        tables.mkdir()
        assert _phase08.fig6_graph_quality_curve(tables, tmp_path / "fig6.pdf") is False
