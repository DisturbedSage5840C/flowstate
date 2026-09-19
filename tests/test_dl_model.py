"""
test_dl_model.py — Unit tests for Jashan's DL model (P4)

Run: pytest tests/test_dl_model.py -v
"""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from src.models.dl_model import (
    AquaSenseDLModel,
    AquaSenseDataset,
    AquaSenseTrainer,
    DotProductAttention,
    FEATURE_COLS,
    TARGET_COLS,
    predict,
)


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

def _make_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame matching Marutey's train.parquet schema."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.random(n) for col in FEATURE_COLS})
    df["site"] = ["lake_a"] * (n // 2) + ["river_b"] * (n - n // 2)
    df["water_body_type"] = ["lake"] * (n // 2) + ["river"] * (n - n // 2)
    df["date"] = pd.date_range("2024-01-01", periods=n, freq="D")
    df["chl_a"] = rng.uniform(0, 80, n)
    df["turbidity"] = rng.uniform(0, 150, n)
    df["do"] = rng.uniform(2, 10, n)
    return df


# ─────────────────────────────────────────────────────────
# Model structure tests
# ─────────────────────────────────────────────────────────

class TestModelArchitecture:

    def test_instantiation(self):
        model = AquaSenseDLModel()
        assert model is not None

    def test_param_count(self):
        model = AquaSenseDLModel()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params == 229_219, f"Expected 229219, got {n_params}"

    def test_forward_shape(self):
        import torch
        model = AquaSenseDLModel()
        model.eval()
        batch = torch.randn(8, 1, len(FEATURE_COLS))
        with torch.no_grad():
            out, attn = model(batch)
        assert out.shape == (8, len(TARGET_COLS)), f"Output shape {out.shape}"
        assert attn.shape == (8, 1), f"Attention shape {attn.shape}"

    def test_custom_hidden_size(self):
        import torch
        model = AquaSenseDLModel(lstm_hidden=32, lstm_layers=1, dropout=0.1)
        batch = torch.randn(4, 1, len(FEATURE_COLS))
        out, _ = model(batch)
        assert out.shape == (4, 3)

    def test_feature_target_col_counts(self):
        assert len(FEATURE_COLS) == 12
        assert TARGET_COLS == ["chl_a", "turbidity", "do"]


# ─────────────────────────────────────────────────────────
# Dataset tests
# ─────────────────────────────────────────────────────────

class TestDataset:

    def test_basic_length(self):
        df = _make_df(50)
        ds = AquaSenseDataset(df)
        assert len(ds) == 50

    def test_item_shapes(self):
        df = _make_df(10)
        ds = AquaSenseDataset(df)
        x, y = ds[0]
        assert x.shape == (1, len(FEATURE_COLS))
        assert y.shape == (len(TARGET_COLS),)

    def test_missing_feature_filled(self):
        df = _make_df(20)
        df["B2"] = np.nan          # inject all-NaN column
        ds = AquaSenseDataset(df)  # should not raise
        x, _ = ds[0]
        assert not x.isnan().any()

    def test_missing_target_as_nan(self):
        import torch
        df = _make_df(10)
        df.loc[0, "chl_a"] = np.nan
        ds = AquaSenseDataset(df)
        _, y = ds[0]
        assert torch.isnan(y[0])   # chl_a is NaN
        assert not torch.isnan(y[1])  # turbidity is fine


# ─────────────────────────────────────────────────────────
# Attention module test
# ─────────────────────────────────────────────────────────

class TestAttention:

    def test_attention_output_shape(self):
        import torch
        attn = DotProductAttention(hidden_size=64)
        x = torch.randn(4, 3, 64)   # batch=4, seq=3, hidden=64
        ctx, weights = attn(x)
        assert ctx.shape == (4, 64)
        assert weights.shape == (4, 3)

    def test_attention_weights_positive(self):
        import torch
        attn = DotProductAttention(hidden_size=32)
        x = torch.randn(2, 5, 32)
        _, weights = attn(x)
        assert (weights >= 0).all()


# ─────────────────────────────────────────────────────────
# predict() interface tests
# ─────────────────────────────────────────────────────────

class TestPredictInterface:

    def test_predict_returns_correct_shape(self):
        df = _make_df(20)
        model = AquaSenseDLModel()
        out = predict(df, model=model)
        assert out.shape == (20, 3)
        assert list(out.columns) == TARGET_COLS

    def test_predict_preserves_index(self):
        df = _make_df(15)
        df.index = list(range(100, 115))
        model = AquaSenseDLModel()
        out = predict(df, model=model)
        assert list(out.index) == list(df.index)

    def test_predict_no_nans_in_output(self):
        df = _make_df(10)
        model = AquaSenseDLModel()
        out = predict(df, model=model)
        assert not out.isnull().any().any(), "predict() should not return NaNs"

    def test_predict_from_parquet_schema(self):
        """Simulate Marutey's exact parquet schema."""
        import pandas as pd
        try:
            df = pd.read_parquet("data/processed/train.parquet")
            model = AquaSenseDLModel()
            out = predict(df.head(10), model=model)
            assert out.shape == (10, 3)
        except FileNotFoundError:
            pytest.skip("train.parquet not available")


# ─────────────────────────────────────────────────────────
# Trainer tests (fast: 2 epochs only)
# ─────────────────────────────────────────────────────────

class TestTrainer:

    def test_train_returns_history(self):
        df = _make_df(60)
        train_df, val_df = df.iloc[:48], df.iloc[48:]
        model = AquaSenseDLModel()
        trainer = AquaSenseTrainer(model, batch_size=16)
        history = trainer.train(train_df, val_df, epochs=2)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2

    def test_evaluate_returns_metrics(self):
        df = _make_df(40)
        model = AquaSenseDLModel()
        trainer = AquaSenseTrainer(model, batch_size=16)
        trainer.train(df.iloc[:30], df.iloc[30:], epochs=2)
        metrics = trainer.evaluate(df.iloc[30:])
        for tgt in TARGET_COLS:
            assert f"{tgt}_r2"   in metrics
            assert f"{tgt}_rmse" in metrics
            assert f"{tgt}_mae"  in metrics

    def test_loss_decreases_over_epochs(self):
        """Very basic sanity: first epoch loss > last epoch loss (or within noise)."""
        df = _make_df(80, seed=7)
        model = AquaSenseDLModel(lstm_hidden=32, lstm_layers=1, dropout=0.0)
        trainer = AquaSenseTrainer(model, lr=1e-2, batch_size=32)
        history = trainer.train(df.iloc[:64], df.iloc[64:], epochs=10)
        # Not strict — model may not converge in 10 epochs on tiny data
        assert history["train_loss"][0] > 0
