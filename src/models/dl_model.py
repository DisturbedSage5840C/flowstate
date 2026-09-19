"""
1D-CNN -> BiLSTM -> masked dot-product attention over a station's visit history.

What is different from a plain per-row network
----------------------------------------------
* **Real temporal input.** Each sample is a window of up to ``window`` satellite visits of the SAME
  station (oldest -> newest, the visit being predicted is last), built from visits no older than
  ``max_gap_days``. A station with no earlier visit gets a window of length 1. Only earlier
  *satellite features* are used, never earlier labels, so nothing about the target leaks in.
  Each step also carries the (log-scaled) time gap to the predicted visit.
* **Attention that can be inspected.** The query is the current visit's BiLSTM state; keys/values are
  the whole window. Padded steps are masked out, so the returned weights are a real distribution over
  the station's history (they are exactly 1.0 only when there is no history).
* **Honest scaling.** Features are median-imputed + standardised with statistics from the training
  data; skewed targets (BOD, turbidity, Chl-a) are modelled on a log1p scale and every target is
  z-scored, so one target's units cannot dominate the loss.
* **No silent bad inputs.** A missing feature column raises ``KeyError``; a feature that is entirely
  missing in the training data raises ``ValueError``; loading a missing artifact raises
  ``FileNotFoundError``. The scalers travel inside the saved artifact.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from src.models.schema import LOG_TARGETS
from src.models.schema import SYNTH_FEATURE_COLS as FEATURE_COLS   # legacy defaults (synthetic demo table)
from src.models.schema import SYNTH_TARGET_COLS as TARGET_COLS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "reports" / "real" / "dl_artifact.pt"


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

class FeatureScaler:
    """Median imputation + standardisation, fitted on training data only."""

    def __init__(self, cols: Sequence[str]):
        self.cols = list(cols)
        self.median: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            raise KeyError(f"missing feature column(s): {missing}")
        return df[self.cols].to_numpy(dtype=np.float64)

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        x = self._matrix(df)
        empty = [c for c, all_nan in zip(self.cols, np.isnan(x).all(axis=0)) if all_nan]
        if empty:
            raise ValueError(f"feature(s) entirely missing in the training data: {empty}")
        self.median = np.nanmedian(x, axis=0)
        filled = np.where(np.isnan(x), self.median, x)
        self.mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.median is None:
            raise RuntimeError("FeatureScaler is not fitted")
        x = self._matrix(df)
        x = np.where(np.isnan(x), self.median, x)
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"cols": self.cols, "median": self.median.tolist(), "mean": self.mean.tolist(),
                "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        s = cls(d["cols"])
        s.median, s.mean, s.std = (np.asarray(d[k], dtype=np.float64) for k in ("median", "mean", "std"))
        return s


class TargetTransform:
    """log1p for skewed targets, then z-score. NaN labels stay NaN."""

    def __init__(self, cols: Sequence[str], log_targets: Sequence[str] = LOG_TARGETS):
        self.cols = list(cols)
        self.log = [c in set(log_targets) for c in self.cols]
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _forward(self, y: np.ndarray) -> np.ndarray:
        out = y.astype(np.float64).copy()
        for i, is_log in enumerate(self.log):
            if is_log:
                out[:, i] = np.log1p(np.maximum(out[:, i], 0.0))
        return out

    def fit(self, df: pd.DataFrame) -> "TargetTransform":
        t = self._forward(df[self.cols].to_numpy(dtype=np.float64))
        for i, c in enumerate(self.cols):
            if np.isnan(t[:, i]).all():
                raise ValueError(f"target {c!r} has no labelled rows in the training data")
        self.mean = np.nanmean(t, axis=0)
        std = np.nanstd(t, axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if all(c in df.columns for c in self.cols):
            y = df[self.cols].to_numpy(dtype=np.float64)
        else:
            y = np.full((len(df), len(self.cols)), np.nan)
        return ((self._forward(y) - self.mean) / self.std).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        t = np.asarray(z, dtype=np.float64) * self.std + self.mean
        for i, is_log in enumerate(self.log):
            if is_log:
                t[:, i] = np.maximum(np.expm1(t[:, i]), 0.0)
        return t

    def to_dict(self) -> dict:
        return {"cols": self.cols, "log": self.log, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetTransform":
        t = cls(d["cols"], log_targets=[c for c, is_log in zip(d["cols"], d["log"]) if is_log])
        t.mean, t.std = np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64)
        return t


# ---------------------------------------------------------------------------
# Visit windows
# ---------------------------------------------------------------------------

def build_windows(df: pd.DataFrame, window: int, max_gap_days: float,
                  site_col: str = "site", date_col: str = "date") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row positions of each sample's window.

    Returns ``(idx, days_before, lengths)``: ``idx`` is (n, window) with valid steps first, ordered
    oldest -> newest (the sample's own row is the last valid step) and -1 for padding;
    ``days_before[i, k]`` is how many days step k precedes the sample's own visit.
    """
    n = len(df)
    idx = np.full((n, window), -1, dtype=np.int64)
    days_before = np.zeros((n, window), dtype=np.float64)
    lengths = np.ones(n, dtype=np.int64)
    if window == 1 or site_col not in df.columns or date_col not in df.columns:
        idx[:, 0] = np.arange(n)
        return idx, days_before, lengths

    dates = pd.to_datetime(df[date_col])
    if dates.isna().any():
        raise ValueError("'date' contains missing values; windows need a date for every row")
    day_num = dates.to_numpy("datetime64[D]").astype(np.int64)
    sites = df[site_col].to_numpy()
    for _, positions in pd.Series(np.arange(n)).groupby(sites):
        pos = positions.to_numpy()
        pos = pos[np.argsort(day_num[pos], kind="stable")]
        d = day_num[pos]
        for k, i in enumerate(pos):
            history: List[int] = []
            j = k - 1
            while j >= 0 and len(history) < window - 1:
                gap = d[k] - d[j]
                if gap > max_gap_days:
                    break
                if gap > 0:                      # strictly earlier visit (same-day duplicates are skipped)
                    history.append(int(pos[j]))
                j -= 1
            seq = history[::-1] + [int(i)]
            L = len(seq)
            lengths[i] = L
            idx[i, :L] = seq
            days_before[i, :L] = [day_num[i] - day_num[q] for q in seq]
    return idx, days_before, lengths


class VisitWindowDataset(Dataset):
    """Samples of shape (window, n_features + 1): scaled features plus a scaled time-gap channel."""

    def __init__(self, df: pd.DataFrame, feature_scaler: FeatureScaler, target_tf: TargetTransform,
                 window: int = 3, max_gap_days: float = 90.0):
        self.window = window
        self.max_gap_days = float(max_gap_days)
        self.features = feature_scaler.transform(df)                      # (n, F)
        self.targets = target_tf.transform(df)                            # (n, K)
        self.idx, self.days_before, self.lengths = build_windows(df, window, max_gap_days)

    @property
    def n_channels(self) -> int:
        return self.features.shape[1] + 1

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int):
        rows = self.idx[i]
        mask = rows >= 0
        x = np.zeros((self.window, self.n_channels), dtype=np.float32)
        x[mask, :-1] = self.features[rows[mask]]
        x[mask, -1] = np.log1p(self.days_before[i][mask]) / math.log1p(self.max_gap_days)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(self.targets[i])


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class MaskedDotProductAttention(nn.Module):
    """Attention of the current visit's state over the window; padded steps get zero weight."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = math.sqrt(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, states: torch.Tensor, mask: torch.Tensor, last_idx: torch.Tensor):
        batch = torch.arange(states.size(0), device=states.device)
        q = self.query(states[batch, last_idx]).unsqueeze(1)                 # (B, 1, H)
        scores = (q * self.key(states)).sum(-1) / self.scale                 # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), states).squeeze(1)         # (B, H)
        return context, weights


class AquaSenseDLModel(nn.Module):
    """Parallel Conv1d (k=1,3,5) -> BiLSTM -> masked attention -> dense regression head."""

    def __init__(self, n_features: int = len(FEATURE_COLS) + 1, lstm_hidden: int = 32, lstm_layers: int = 1,
                 dropout: float = 0.2, n_targets: int = 3, conv_out: int = 16):
        super().__init__()
        self.n_features, self.n_targets = n_features, n_targets
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(n_features, conv_out, kernel_size=k, padding=k // 2),
                          nn.GroupNorm(1, conv_out), nn.ReLU())
            for k in (1, 3, 5)
        ])
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(3 * conv_out, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        h2 = 2 * lstm_hidden
        self.attention = MaskedDotProductAttention(h2)
        self.head = nn.Sequential(nn.Linear(2 * h2, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_targets))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """x: (B, T, n_features); mask: (B, T) bool, True = real step. Returns (out, attention (B, T))."""
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        m = mask.unsqueeze(-1).to(x.dtype)
        conv_in = (x * m).transpose(1, 2)                                     # (B, C, T)
        conv = torch.cat([c(conv_in) for c in self.convs], dim=1)             # (B, 3*co, T)
        seq = self.dropout(conv).transpose(1, 2) * m                          # (B, T, 3*co)
        lengths = mask.sum(1).clamp(min=1)
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(out, batch_first=True, total_length=T)  # (B, T, 2H)
        last_idx = lengths - 1
        context, weights = self.attention(states, mask, last_idx)
        current = states[torch.arange(B, device=x.device), last_idx]
        return self.head(torch.cat([context, current], dim=-1)), weights


# ---------------------------------------------------------------------------
# Predictor (model + scalers + config travel together)
# ---------------------------------------------------------------------------

@dataclass
class DLPredictor:
    model: AquaSenseDLModel
    feature_scaler: FeatureScaler
    target_tf: TargetTransform
    window: int = 3
    max_gap_days: float = 90.0
    model_kwargs: dict | None = None

    @property
    def feature_cols(self) -> List[str]:
        return self.feature_scaler.cols

    @property
    def target_cols(self) -> List[str]:
        return self.target_tf.cols

    def predict(self, df: pd.DataFrame, return_attention: bool = False, device: str = "cpu"):
        """Predictions in original units, indexed like ``df``. History windows are built inside ``df``."""
        ds = VisitWindowDataset(df, self.feature_scaler, self.target_tf, self.window, self.max_gap_days)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        self.model.to(device).eval()
        preds, attn = [], []
        with torch.no_grad():
            for x, mask, _ in loader:
                out, w = self.model(x.to(device), mask.to(device))
                preds.append(out.cpu().numpy())
                attn.append(w.cpu().numpy())
        z = np.vstack(preds) if preds else np.empty((0, len(self.target_cols)))
        result = pd.DataFrame(self.target_tf.inverse(z), columns=self.target_cols, index=df.index)
        return (result, np.vstack(attn) if attn else np.empty((0, self.window))) if return_attention else result

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "model_kwargs": self.model_kwargs or {},
            "window": self.window, "max_gap_days": self.max_gap_days,
            "feature_scaler": self.feature_scaler.to_dict(), "target_tf": self.target_tf.to_dict(),
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT) -> "DLPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No DL artifact at {path}. Train one with `python -m scripts.train_dl_real` "
                "(real data) or `python scripts/train_dl_model.py` (synthetic demo).")
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = AquaSenseDLModel(**blob["model_kwargs"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return cls(model, FeatureScaler.from_dict(blob["feature_scaler"]), TargetTransform.from_dict(blob["target_tf"]),
                   blob["window"], blob["max_gap_days"], blob["model_kwargs"])


def predict(df: pd.DataFrame, artifact_path: str | Path = DEFAULT_ARTIFACT) -> pd.DataFrame:
    """Module-level handoff interface. Raises FileNotFoundError when no trained artifact exists."""
    return DLPredictor.load(artifact_path).predict(df)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class AquaSenseTrainer:
    """Fits scalers on the training data, trains with early stopping, returns a DLPredictor."""

    def __init__(self, feature_cols: Sequence[str] = FEATURE_COLS, target_cols: Sequence[str] = TARGET_COLS,
                 window: int = 3, max_gap_days: float = 90.0, lr: float = 2e-3, weight_decay: float = 1e-4,
                 batch_size: int = 64, hidden: int = 32, layers: int = 1, dropout: float = 0.2, conv_out: int = 16,
                 target_weights: Optional[Dict[str, float]] = None, seed: int = 0, device: Optional[str] = None):
        self.feature_cols, self.target_cols = list(feature_cols), list(target_cols)
        self.window, self.max_gap_days = window, max_gap_days
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.model_kwargs = {"n_features": len(self.feature_cols) + 1, "lstm_hidden": hidden, "lstm_layers": layers,
                             "dropout": dropout, "n_targets": len(self.target_cols), "conv_out": conv_out}
        w = target_weights or {}
        self.weights = torch.tensor([w.get(c, 1.0) for c in self.target_cols], dtype=torch.float32)
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history: Dict[str, List[float]] = {}

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(target)
        if not valid.any():
            return pred.sum() * 0.0
        err = F.huber_loss(pred, torch.nan_to_num(target), reduction="none", delta=1.0)
        w = self.weights.to(pred.device).unsqueeze(0).expand_as(err)
        return (err * w * valid).sum() / (w * valid).sum().clamp(min=1e-8)

    def _epoch(self, model, loader, opt=None) -> float:
        train = opt is not None
        model.train(train)
        total, count = 0.0, 0
        with torch.set_grad_enabled(train):
            for x, mask, y in loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device)
                pred, _ = model(x, mask)
                loss = self._loss(pred, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                n_valid = int((~torch.isnan(y)).sum())
                total += float(loss.detach()) * n_valid
                count += n_valid
        return total / max(count, 1)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, epochs: int = 80,
            patience: int = 15, fixed_epochs: bool = False) -> DLPredictor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        fscaler = FeatureScaler(self.feature_cols).fit(train_df)
        ttf = TargetTransform(self.target_cols).fit(train_df)
        train_ds = VisitWindowDataset(train_df, fscaler, ttf, self.window, self.max_gap_days)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = None
        if val_df is not None and len(val_df):
            val_ds = VisitWindowDataset(val_df, fscaler, ttf, self.window, self.max_gap_days)
            val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        model = AquaSenseDLModel(**self.model_kwargs).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.history = {"train_loss": [], "val_loss": []}
        best, best_state, best_epoch, wait = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
        for epoch in range(1, epochs + 1):
            tr = self._epoch(model, train_loader, opt)
            self.history["train_loss"].append(tr)
            monitored = tr
            if val_loader is not None:
                va = self._epoch(model, val_loader)
                self.history["val_loss"].append(va)
                monitored = va
            if monitored < best - 1e-5:
                best, best_state, best_epoch, wait = monitored, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                wait += 1
                if not fixed_epochs and wait >= patience:
                    break
        if not fixed_epochs:
            model.load_state_dict(best_state)
        self.best_epoch = best_epoch
        model.cpu().eval()
        return DLPredictor(model, fscaler, ttf, self.window, self.max_gap_days, dict(self.model_kwargs))


def group_holdout(df: pd.DataFrame, frac: float = 0.15, seed: int = 0, site_col: str = "site"):
    """Split by whole stations (never by rows) -> (train_part, holdout_part)."""
    sites = np.array(sorted(df[site_col].unique()))
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(frac * len(sites)))) if len(sites) > 1 else 0
    hold = set(rng.choice(sites, size=n_hold, replace=False)) if n_hold else set()
    mask = df[site_col].isin(hold)
    return df[~mask], df[mask]


def oof_predictions(df: pd.DataFrame, feature_cols: Sequence[str], target_cols: Sequence[str], n_folds: int = 5,
                    seed: int = 42, epochs: int = 80, patience: int = 15, **trainer_kw) -> Tuple[pd.DataFrame, List[dict]]:
    """Out-of-fold predictions on the SAME site-blocked spatial folds the XGBoost pipeline uses."""
    from src.models.spatial_cv import SpatialKFold

    df = df.reset_index(drop=True)
    oof = pd.DataFrame(np.nan, index=df.index, columns=list(target_cols))
    info: List[dict] = []
    for fold, (train_idx, val_idx) in enumerate(SpatialKFold(n_folds=n_folds, random_state=seed).split(df)):
        train_part, stop_part = group_holdout(df.iloc[train_idx], frac=0.15, seed=seed + fold)
        trainer = AquaSenseTrainer(feature_cols, target_cols, seed=seed + fold, **trainer_kw)
        predictor = trainer.fit(train_part, stop_part, epochs=epochs, patience=patience)
        oof.iloc[val_idx] = predictor.predict(df.iloc[val_idx]).to_numpy()
        info.append({"fold": fold, "n_train": int(len(train_part)), "n_early_stop": int(len(stop_part)),
                     "n_heldout": int(len(val_idx)), "best_epoch": int(trainer.best_epoch)})
    return oof, info
