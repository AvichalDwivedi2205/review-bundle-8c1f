from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_frozen_paper_bundle_health_checks_pass():
    health_path = Path("outputs/paper-freeze-2026-04-10/tables/benchmark_health.csv")
    if not health_path.exists():
        pytest.skip("paper freeze outputs are not present in this checkout")

    health = pd.read_csv(health_path)

    assert not health.empty
    assert health["passes"].astype(bool).all()
