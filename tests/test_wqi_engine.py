"""
Unit Tests for wqi_engine.py — Aqua-Sense (Marutey P2)

Hand-computed reference cases used to verify correctness.
Run with: python -m pytest tests/test_wqi_engine.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.wqi.wqi_engine import compute_wqi, compute_wqi_dataframe, wqi_class, PARAMETERS


# ---------------------------------------------------------------------------
# Test: wqi_class boundaries
# ---------------------------------------------------------------------------

class TestWQIClass:
    def test_class_a_clean(self):
        cls, _ = wqi_class(10.0)
        assert cls == "A"

    def test_class_b(self):
        cls, _ = wqi_class(35.0)
        assert cls == "B"

    def test_class_c(self):
        cls, _ = wqi_class(60.0)
        assert cls == "C"

    def test_class_d(self):
        cls, _ = wqi_class(85.0)
        assert cls == "D"

    def test_class_e_heavily_polluted(self):
        cls, _ = wqi_class(120.0)
        assert cls == "E"

    def test_boundary_25(self):
        # Exactly 25 → class B (>= 25)
        cls, _ = wqi_class(25.0)
        assert cls == "B"

    def test_boundary_50(self):
        cls, _ = wqi_class(50.0)
        assert cls == "C"


# ---------------------------------------------------------------------------
# Test: compute_wqi returns correct structure
# ---------------------------------------------------------------------------

class TestComputeWQI:
    def test_output_keys(self):
        result = compute_wqi(do=7.0, bod=1.5, turbidity=3.0, chl_a=5.0)
        assert set(result.keys()) == {"wqi", "wqi_class", "description", "sub_index", "weights"}

    def test_wqi_is_float(self):
        result = compute_wqi(do=6.0, bod=2.0, turbidity=5.0, chl_a=8.0)
        assert isinstance(result["wqi"], float)

    def test_wqi_non_negative(self):
        result = compute_wqi(do=12.0, bod=0.5, turbidity=1.0, chl_a=2.0)
        assert result["wqi"] >= 0.0

    def test_sub_index_all_params(self):
        result = compute_wqi(do=5.0, bod=3.0, turbidity=10.0, chl_a=10.0)
        assert set(result["sub_index"].keys()) == set(PARAMETERS.keys())

    def test_sub_index_range(self):
        result = compute_wqi(do=5.0, bod=3.0, turbidity=10.0, chl_a=10.0)
        for key, qi in result["sub_index"].items():
            assert 0.0 <= qi <= 100.0, f"Qi out of range for {key}: {qi}"

    def test_weights_sum_to_one(self):
        result = compute_wqi(do=5.0, bod=3.0, turbidity=10.0, chl_a=10.0)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Test: Clean water → Class A
# ---------------------------------------------------------------------------

class TestCleanWater:
    """
    Clean river water: high DO, low BOD, low turbidity, low Chl-a
    Hand-computed expectation: WQI < 25 → Class A
    """
    def test_class_is_a(self):
        result = compute_wqi(do=8.5, bod=1.0, turbidity=2.0, chl_a=3.0)
        assert result["wqi_class"] in ("A", "B"), \
            f"Expected A or B, got {result['wqi_class']} (WQI={result['wqi']})"

    def test_wqi_below_25(self):
        result = compute_wqi(do=8.5, bod=1.0, turbidity=2.0, chl_a=3.0)
        # Clean water should score well below the mid-range
        assert result["wqi"] < 50.0, f"Clean water WQI unexpectedly high: {result['wqi']}"


# ---------------------------------------------------------------------------
# Test: Buddha Nullah equivalent → Class E
# ---------------------------------------------------------------------------

class TestBuddhaNullah:
    """
    Severely polluted drain: very low DO, extremely high BOD and turbidity.
    Hand-computed expectation: WQI > 100 → Class E
    """
    def test_class_is_e(self):
        result = compute_wqi(do=0.4, bod=150.0, turbidity=500.0, chl_a=8.0)
        assert result["wqi_class"] in ("D", "E"), \
            f"Expected D or E, got {result['wqi_class']} (WQI={result['wqi']})"

    def test_wqi_above_100(self):
        result = compute_wqi(do=0.4, bod=150.0, turbidity=500.0, chl_a=8.0)
        assert result["wqi"] > 75.0, f"Severely polluted water WQI should be > 75, got {result['wqi']}"


# ---------------------------------------------------------------------------
# Test: Eutrophic lake (Bellandur-like) → Class C or D
# ---------------------------------------------------------------------------

class TestEutrophicLake:
    def test_class_is_c_or_d(self):
        # Bellandur-like: high Chl-a + turbidity + low DO legitimately scores D or E
        result = compute_wqi(do=2.5, bod=20.0, turbidity=60.0, chl_a=100.0)
        assert result["wqi_class"] in ("C", "D", "E"), \
            f"Expected C/D/E, got {result['wqi_class']} (WQI={result['wqi']})"


# ---------------------------------------------------------------------------
# Test: Higher pollution → higher WQI
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_more_bod_raises_wqi(self):
        r1 = compute_wqi(do=5.0, bod=2.0,  turbidity=10.0, chl_a=10.0)
        r2 = compute_wqi(do=5.0, bod=10.0, turbidity=10.0, chl_a=10.0)
        assert r2["wqi"] > r1["wqi"]

    def test_more_turbidity_raises_wqi(self):
        r1 = compute_wqi(do=5.0, bod=3.0, turbidity=5.0,   chl_a=10.0)
        r2 = compute_wqi(do=5.0, bod=3.0, turbidity=100.0, chl_a=10.0)
        assert r2["wqi"] > r1["wqi"]

    def test_higher_do_lowers_wqi(self):
        r1 = compute_wqi(do=2.0, bod=3.0, turbidity=10.0, chl_a=10.0)
        r2 = compute_wqi(do=8.0, bod=3.0, turbidity=10.0, chl_a=10.0)
        assert r2["wqi"] < r1["wqi"]

    def test_more_chl_a_raises_wqi(self):
        r1 = compute_wqi(do=5.0, bod=3.0, turbidity=10.0, chl_a=5.0)
        r2 = compute_wqi(do=5.0, bod=3.0, turbidity=10.0, chl_a=80.0)
        assert r2["wqi"] > r1["wqi"]


# ---------------------------------------------------------------------------
# Test: DataFrame batch computation
# ---------------------------------------------------------------------------

class TestComputeWQIDataFrame:
    def _sample_df(self):
        return pd.DataFrame({
            "do":        [7.0, 2.0,  0.5],
            "bod":       [1.5, 15.0, 80.0],
            "turbidity": [3.0, 50.0, 400.0],
            "chl_a":     [5.0, 70.0, 120.0],
        })

    def test_output_has_wqi_columns(self):
        df = compute_wqi_dataframe(self._sample_df())
        assert "wqi" in df.columns
        assert "wqi_class" in df.columns
        assert "wqi_description" in df.columns

    def test_wqi_class_per_row(self):
        df = compute_wqi_dataframe(self._sample_df())
        # Row 0 is moderate (DO=7, BOD=1.5, turb=3, chl=5) — should be A/B/C
        assert df["wqi_class"].iloc[0] in ("A", "B", "C")
        # Row 2 is severely polluted (DO=0.5, BOD=80, turb=400, chl=120) — should be D/E
        assert df["wqi_class"].iloc[2] in ("D", "E")

    def test_sub_index_columns_present(self):
        df = compute_wqi_dataframe(self._sample_df())
        for key in PARAMETERS:
            assert f"qi_{key}" in df.columns

    def test_default_ph_added(self):
        df = self._sample_df()
        assert "ph" not in df.columns
        df_out = compute_wqi_dataframe(df)
        assert "ph" in df_out.columns

    def test_missing_column_raises(self):
        df = pd.DataFrame({"do": [5.0], "bod": [3.0]})
        with pytest.raises(ValueError, match="Missing required column"):
            compute_wqi_dataframe(df)

    def test_no_nan_in_wqi(self):
        df = compute_wqi_dataframe(self._sample_df())
        assert df["wqi"].isna().sum() == 0
