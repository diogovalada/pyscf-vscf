from __future__ import annotations

import pytest


@pytest.mark.pyscf
@pytest.mark.slow
def test_water_example_runs_and_orders_fundamentals(tmp_path) -> None:
    pytest.importorskip("pyscf")
    from argparse import Namespace

    from pyscf_vscf.examples import water_vscf_vci

    result = water_vscf_vci.run(
        Namespace(
            method="hf",
            basis="sto-3g",
            nodes=5,
            solver_points=15,
            modals=4,
            store=tmp_path / "points.sqlite",
        )
    )

    rows = result["fundamentals"]
    assert [row["mode"] for row in rows] == ["bend", "symmetric stretch", "antisymmetric stretch"]
    for row in rows:
        assert row["vscf_cm"] < row["harmonic_cm"]
        assert row["vscf_km_per_mol"] > 0.0
        assert row["vci_cm"] is not None
        assert abs(row["vci_cm"] - row["vscf_cm"]) < 100.0
