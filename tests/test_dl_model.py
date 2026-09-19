"""Tests for the CNN-BiLSTM-attention model: windows, masking, scaling, artifacts, and that it actually learns."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.dl_model import (
    AquaSenseDLModel,
    AquaSenseTrainer,
    DLPredictor,
    FeatureScaler,
    MaskedDotProductAttention,
    TargetTransform,
    VisitWindowDataset,
    build_windows,
    group_holdout,
    oof_predictions,
    predict,
)

FEATS = ["f1", "f2", "f3"]


def make_df(n_sites=12, visits=8, seed=0, gap_days=20):
    """Learnable task: turbidity depends on f1 and on the PREVIOUS visit's f2 (needs history)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        base = pd.Timestamp("2020-01-01")
        prev_f2 = 0.0
        for v in range(visits):
            f = rng.normal(size=3)
            rows.append({"site": f"S{s}", "date": base + pd.Timedelta(days=gap_days * v),
                         "lat": 10 + s, "lon": 70 + s, "water_body_type": "lake" if s % 2 else "river",
                         "f1": f[0], "f2": f[1], "f3": f[2],
                         "turbidity": float(np.exp(1.0 + 0.8 * f[0] + 0.6 * prev_f2 + rng.normal(0, 0.05))),
                         "do": float(6 + f[2] + rng.normal(0, 0.05))})
            prev_f2 = f[1]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def test_windows_are_chronological_per_site_and_end_at_the_current_visit():
    df = make_df(n_sites=2, visits=5, gap_days=20)
    idx, days_before, lengths = build_windows(df, window=3, max_gap_days=90)
    for i in range(len(df)):
        rows = idx[i][idx[i] >= 0]
        assert rows[-1] == i                                             # current visit is last
        assert (df["site"].to_numpy()[rows] == df["site"].iloc[i]).all() # never crosses stations
        dates = df["date"].to_numpy()[rows]
        assert (np.diff(dates.astype("int64")) > 0).all()                # strictly oldest -> newest
        assert days_before[i][len(rows) - 1] == 0
    assert lengths.max() == 3 and lengths.min() == 1


def test_windows_never_use_later_visits_and_respect_the_gap_limit():
    df = pd.DataFrame({"site": ["a"] * 3, "date": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-06-01"]),
                       "f1": [1.0, 2.0, 3.0]})
    idx, _, lengths = build_windows(df, window=3, max_gap_days=90)
    assert idx[0].tolist() == [0, -1, -1]                # first visit has no history (later visits are not used)
    assert idx[1].tolist() == [0, 1, -1]
    assert idx[2].tolist() == [2, -1, -1]                # previous visit is 143 days away > 90
    assert lengths.tolist() == [1, 2, 1]


def test_same_day_duplicates_are_not_history():
    df = pd.DataFrame({"site": ["a", "a"], "date": pd.to_datetime(["2020-01-01", "2020-01-01"]), "f1": [1.0, 2.0]})
    assert build_windows(df, 3, 90)[2].tolist() == [1, 1]


def test_window_of_one_or_missing_columns_gives_single_step_samples():
    df = make_df(2, 3)
    assert build_windows(df, 1, 90)[2].tolist() == [1] * len(df)
    assert build_windows(df.drop(columns=["site"]), 3, 90)[2].tolist() == [1] * len(df)


def test_missing_dates_raise():
    df = make_df(1, 3)
    df.loc[1, "date"] = pd.NaT
    with pytest.raises(ValueError, match="date"):
        build_windows(df, 3, 90)


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

def test_feature_scaler_fills_nan_with_training_median_and_standardises():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, np.nan], "f2": [10.0, 10.0, 10.0, 10.0]})
    sc = FeatureScaler(["f1", "f2"]).fit(df)
    out = sc.transform(df)
    assert np.isfinite(out).all()
    assert out[3, 0] == pytest.approx(0.0, abs=1e-6)                   # NaN -> median (2.0) -> z = 0
    assert np.allclose(out[:, 1], 0.0)                                # constant column does not blow up


def test_missing_or_fully_empty_features_raise_instead_of_becoming_zeros():
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="f2"):
        FeatureScaler(["f1", "f2"]).fit(df)
    sc = FeatureScaler(["f1"]).fit(df)
    with pytest.raises(KeyError, match="f1"):
        sc.transform(pd.DataFrame({"other": [1.0]}))


def test_target_transform_roundtrip_log_and_identity_and_nan():
    df = pd.DataFrame({"bod": [1.0, 10.0, 100.0, np.nan], "do": [2.0, 5.0, 8.0, 6.0]})
    tt = TargetTransform(["bod", "do"]).fit(df)
    z = tt.transform(df)
    assert np.isnan(z[3, 0]) and np.isfinite(z[3, 1])
    assert np.nanmean(z[:, 0]) == pytest.approx(0.0, abs=1e-5)         # z-scored on the log scale
    back = tt.inverse(np.nan_to_num(z))
    assert back[:3, 0] == pytest.approx([1.0, 10.0, 100.0], rel=1e-4)
    assert back[:, 1] == pytest.approx([2.0, 5.0, 8.0, 6.0], rel=1e-4)


def test_target_without_labels_raises():
    with pytest.raises(ValueError, match="bod"):
        TargetTransform(["bod"]).fit(pd.DataFrame({"bod": [np.nan, np.nan]}))


# ---------------------------------------------------------------------------
# Network and attention
# ---------------------------------------------------------------------------

def test_forward_shapes_and_attention_is_a_distribution_over_real_steps_only():
    model = AquaSenseDLModel(n_features=4, n_targets=2).eval()
    x = torch.randn(5, 3, 4)
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool)
    out, w = model(x, mask)
    assert out.shape == (5, 2) and w.shape == (5, 3)
    assert torch.allclose(w.sum(1), torch.ones(5), atol=1e-5)
    assert (w[~mask] == 0).all()                                          # padded steps get zero weight
    assert torch.allclose(w[0], torch.tensor([1.0, 0.0, 0.0]))            # no history -> exactly 1.0 (and only then)
    x2 = x.clone()
    x2[2, 0] += 5.0                                                      # change the content of an old visit
    _, w2 = model(x2, mask)
    assert not torch.allclose(w[2], w2[2], atol=1e-7)                     # weights react to the history's content


def test_padding_content_cannot_influence_the_output():
    model = AquaSenseDLModel(n_features=4, n_targets=1).eval()
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    out1, _ = model(x, mask)
    x2 = x.clone()
    x2[0, 2] = 99.0
    x2[1, 1:] = -50.0                                                      # garbage in the padded slots
    out2, _ = model(x2, mask)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_single_sample_batch_works_in_train_mode():
    model = AquaSenseDLModel(n_features=4, n_targets=1).train()
    out, w = model(torch.randn(1, 1, 4), torch.ones(1, 1, dtype=torch.bool))
    assert out.shape == (1, 1) and w.item() == pytest.approx(1.0)


def test_attention_module_masks_padding():
    att = MaskedDotProductAttention(8)
    states = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    ctx, w = att(states, mask, mask.sum(1) - 1)
    assert ctx.shape == (2, 8) and (w[0, 2:] == 0).all() and w[1].sum().item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_dataset_item_layout():
    df = make_df(2, 4)
    sc, tt = FeatureScaler(FEATS).fit(df), TargetTransform(["turbidity", "do"]).fit(df)
    ds = VisitWindowDataset(df, sc, tt, window=3, max_gap_days=90)
    x, mask, y = ds[2]                                      # third visit of site S0: two earlier visits
    assert x.shape == (3, len(FEATS) + 1) and mask.tolist() == [True, True, True] and y.shape == (2,)
    assert x[-1, -1] == 0.0 and x[0, -1] > x[1, -1] > 0     # time-gap channel: 0 for the current visit, larger further back
    x0, m0, _ = ds[0]
    assert m0.tolist() == [True, False, False] and (x0[1:] == 0).all()


# ---------------------------------------------------------------------------
# Training, prediction, artifacts
# ---------------------------------------------------------------------------

def test_training_learns_and_beats_the_mean_baseline():
    df = make_df(n_sites=16, visits=10, seed=1)
    train, test = df[df["site"].isin([f"S{i}" for i in range(12)])], df[~df["site"].isin([f"S{i}" for i in range(12)])]
    tr, stop = group_holdout(train, frac=0.2, seed=0)
    trainer = AquaSenseTrainer(FEATS, ["turbidity", "do"], window=3, hidden=16, conv_out=8, seed=0, lr=5e-3)
    pred = trainer.fit(tr, stop, epochs=60, patience=60)
    h = trainer.history["train_loss"]
    assert h[-1] < 0.6 * h[0]                                # loss falls substantially
    out = pred.predict(test)
    for t in ("turbidity", "do"):
        y = test[t].to_numpy()
        ss_res = ((y - out[t].to_numpy()) ** 2).sum()
        ss_tot = ((y - train[t].mean()) ** 2).sum()
        assert 1 - ss_res / ss_tot > 0.3, t                  # held-out stations: clearly better than the training mean
    assert list(out.index) == list(test.index) and (out >= 0).all().all()


def test_history_window_helps_on_a_task_that_needs_it():
    df = make_df(n_sites=16, visits=10, seed=2)
    tr_sites = [f"S{i}" for i in range(12)]
    train, test = df[df["site"].isin(tr_sites)], df[~df["site"].isin(tr_sites)]
    scores = {}
    for window in (1, 3):
        tr, stop = group_holdout(train, 0.2, 0)
        p = AquaSenseTrainer(FEATS, ["turbidity"], window=window, hidden=16, conv_out=8, seed=0, lr=5e-3) \
            .fit(tr, stop, epochs=80, patience=80)
        y, yhat = np.log1p(test["turbidity"].to_numpy()), np.log1p(p.predict(test)["turbidity"].to_numpy())
        scores[window] = np.mean((y - yhat) ** 2)
    assert scores[3] < scores[1]                              # the previous visit carries real signal


def test_predictor_index_alignment_and_missing_column_error():
    df = make_df(6, 5)
    p = AquaSenseTrainer(FEATS, ["do"], hidden=8, conv_out=4, seed=0).fit(df, None, epochs=3, fixed_epochs=True)
    shuffled = df.sample(frac=1, random_state=0)
    assert list(p.predict(shuffled).index) == list(shuffled.index)
    with pytest.raises(KeyError):
        p.predict(df.drop(columns=["f2"]))


def test_artifact_roundtrip_reproduces_predictions_and_missing_artifact_raises(tmp_path):
    df = make_df(6, 5)
    p = AquaSenseTrainer(FEATS, ["turbidity", "do"], hidden=8, conv_out=4, seed=0).fit(df, None, epochs=4, fixed_epochs=True)
    path = p.save(tmp_path / "a.pt")
    loaded = DLPredictor.load(path)
    pd.testing.assert_frame_equal(p.predict(df), loaded.predict(df), rtol=1e-5, atol=1e-5)
    assert loaded.feature_cols == FEATS and loaded.target_cols == ["turbidity", "do"]
    with pytest.raises(FileNotFoundError, match="No DL artifact"):
        predict(df, artifact_path=tmp_path / "missing.pt")


def test_oof_predictions_use_site_blocked_folds_and_cover_every_row():
    df = make_df(n_sites=10, visits=6, seed=3)
    oof, info = oof_predictions(df, FEATS, ["do"], n_folds=3, seed=0, epochs=4, patience=4, hidden=8, conv_out=4)
    assert oof["do"].notna().all() and len(oof) == len(df)
    assert sum(f["n_heldout"] for f in info) == len(df)
    assert all(f["n_train"] > 0 and f["n_early_stop"] > 0 for f in info)


def test_group_holdout_splits_by_station():
    df = make_df(10, 4)
    a, b = group_holdout(df, frac=0.3, seed=1)
    assert set(a["site"]).isdisjoint(set(b["site"])) and len(a) + len(b) == len(df)
