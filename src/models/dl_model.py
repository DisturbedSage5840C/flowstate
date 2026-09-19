"""
dl_model.py — AquaSense PyTorch 1D-CNN-BiLSTM-Attention model (Jashan / P4)

Architecture:
    Parallel Conv1d (k=1,3,5) → concat → BiLSTM → dot-product Attention → Dense head
    Outputs: chl_a, turbidity, do  (regression, 3 heads)

Exposes:
    AquaSenseDLModel   — nn.Module
    AquaSenseTrainer   — train / eval loop with spatial-kfold support
    predict(df)        — same interface as XGBoost model (Navya's spatial_cv.py)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────
# Feature schema (matches Marutey's train.parquet handoff)
# ─────────────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    "B2", "B3", "B4", "B5", "B8", "B11",       # Sentinel-2 bands
    "ndci",                                      # Normalised Difference Chlorophyll Index
    "bdm2", "bdm3",                             # 2-band / 3-band difference models
    "red_green",                                 # red/green ratio
    "nir",                                       # NIR reflectance
    "temp_surface",                              # surface temperature proxy
]
TARGET_COLS: List[str] = ["chl_a", "turbidity", "do"]

# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────

class AquaSenseDataset(Dataset):
    """
    Wraps a DataFrame of water-quality observations into a PyTorch Dataset.

    Each sample is a feature vector (1D, length = len(FEATURE_COLS)).
    The Conv1d treats each feature as a "channel" over a sequence of length 1
    unless we stack temporal windows (see `window_size`).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain FEATURE_COLS + TARGET_COLS.
    window_size : int
        Number of consecutive time steps per sample. 1 = point-in-time mode.
    scaler : optional sklearn-style scaler applied to features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 1,
        scaler=None,
        fit_scaler: bool = False,
    ):
        self.window_size = window_size

        # Fill missing features with column median
        df = df.copy()
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)
        for col in TARGET_COLS:
            if col not in df.columns:
                df[col] = np.nan

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[TARGET_COLS].values.astype(np.float32)

        if scaler is not None:
            if fit_scaler:
                scaler.fit(X)
            X = scaler.transform(X)

        self.X = X          # (N, n_features)
        self.y = y          # (N, 3)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.X[idx], dtype=torch.float32)          # (n_features,)
        y = torch.tensor(self.y[idx], dtype=torch.float32)           # (3,)
        # Reshape to (seq_len=1, n_features) for Conv1d input (batch, channels, L)
        x = x.unsqueeze(0)                                           # (1, n_features)
        return x, y


# ─────────────────────────────────────────────────────────
# Attention module
# ─────────────────────────────────────────────────────────

class DotProductAttention(nn.Module):
    """
    Dot-product (scaled) self-attention over the LSTM time dimension.
    Input:  (batch, seq_len, hidden_size)
    Output: context vector (batch, hidden_size)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = hidden_size ** 0.5
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key   = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_out: (batch, seq, hidden)
        Q = self.query(lstm_out)                                # (batch, seq, hidden)
        K = self.key(lstm_out)                                  # (batch, seq, hidden)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, seq, seq)
        weights = F.softmax(scores, dim=-1)                     # (batch, seq, seq)
        context = torch.bmm(weights, lstm_out)                  # (batch, seq, hidden)
        # Mean-pool over seq to get fixed context
        context = context.mean(dim=1)                           # (batch, hidden)
        attn_weights = weights.mean(dim=1)                      # (batch, seq) for vis
        return context, attn_weights


# ─────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────

class AquaSenseDLModel(nn.Module):
    """
    Parallel Conv1d → BiLSTM → DotProductAttention → Dense (3-head regression)

    Parameters
    ----------
    n_features   : number of input spectral/derived features (default: len(FEATURE_COLS))
    lstm_hidden  : BiLSTM hidden units per direction
    lstm_layers  : number of stacked LSTM layers
    dropout      : dropout probability (applied after LSTM and in dense head)
    n_targets    : 3 (chl_a, turbidity, do)
    """

    def __init__(
        self,
        n_features: int = len(FEATURE_COLS),
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        n_targets: int = 3,
    ):
        super().__init__()
        self.n_features  = n_features
        self.lstm_hidden = lstm_hidden
        self.n_targets   = n_targets

        # ── Parallel Conv1d branches (k=1, 3, 5) ──────────────
        # Conv1d expects (batch, C_in, L); we treat features as C_in, seq_len as L
        conv_out = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, conv_out, kernel_size=1, padding=0),
            nn.BatchNorm1d(conv_out),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(n_features, conv_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(n_features, conv_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_out),
            nn.ReLU(),
        )
        self.conv_dropout = nn.Dropout(dropout)

        # After concat: 3 * conv_out channels over the seq
        cnn_out_channels = 3 * conv_out

        # ── BiLSTM ────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_size = lstm_hidden * 2   # bidirectional

        # ── Attention ─────────────────────────────────────────
        self.attention = DotProductAttention(hidden_size=lstm_out_size)

        # ── Dense head ────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_targets),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)
            seq_len=1 for point-in-time; >1 for temporal windows.

        Returns
        -------
        out          : (batch, n_targets)
        attn_weights : (batch, seq_len) — for interpretability
        """
        batch, seq_len, n_feat = x.shape

        # Conv1d expects (batch, channels, length) → transpose feat/seq
        xc = x.transpose(1, 2)           # (batch, n_features, seq_len)

        c1 = self.conv1(xc)              # (batch, 32, seq_len)
        c3 = self.conv3(xc)              # (batch, 32, seq_len)
        c5 = self.conv5(xc)              # (batch, 32, seq_len)

        cnn_out = torch.cat([c1, c3, c5], dim=1)   # (batch, 96, seq_len)
        cnn_out = self.conv_dropout(cnn_out)

        # BiLSTM expects (batch, seq_len, channels)
        lstm_in = cnn_out.transpose(1, 2)            # (batch, seq_len, 96)
        lstm_out, _ = self.lstm(lstm_in)             # (batch, seq_len, 2*hidden)

        # Attention
        context, attn_weights = self.attention(lstm_out)  # (batch, 2*hidden)

        # Regression head
        out = self.head(context)                          # (batch, n_targets)
        return out, attn_weights


# ─────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────

class AquaSenseTrainer:
    """
    Train/eval loop for AquaSenseDLModel.

    Supports spatial-kfold by accepting pre-split DataFrames from Navya's
    spatial_cv.py fold assignment.

    Usage
    -----
    trainer = AquaSenseTrainer(model)
    trainer.train(train_df, val_df, epochs=50)
    metrics = trainer.evaluate(test_df)
    """

    def __init__(
        self,
        model: AquaSenseDLModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        device: Optional[str] = None,
        scaler=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.scaler = scaler
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.criterion = nn.HuberLoss(delta=1.0)  # robust to outlier observations

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_r2": []
        }

    def _make_loader(self, df: pd.DataFrame, shuffle: bool, fit_scaler: bool = False) -> DataLoader:
        ds = AquaSenseDataset(df, scaler=self.scaler, fit_scaler=fit_scaler)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device == "cuda"),
        )

    def _step(self, loader: DataLoader, train: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        total_loss = 0.0
        preds_list, targets_list = [], []

        with torch.set_grad_enabled(train):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)     # (B, 1, n_feat)
                y_batch = y_batch.to(self.device)     # (B, 3)

                pred, _ = self.model(X_batch)

                # Mask NaN targets (missing ground truth for some parameters)
                mask = ~torch.isnan(y_batch)
                if mask.sum() == 0:
                    continue
                loss = self.criterion(pred[mask], y_batch[mask])

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * mask.sum().item()
                preds_list.append(pred.detach().cpu().numpy())
                targets_list.append(y_batch.detach().cpu().numpy())

        preds   = np.vstack(preds_list)   if preds_list   else np.array([])
        targets = np.vstack(targets_list) if targets_list else np.array([])
        avg_loss = total_loss / max(len(loader.dataset), 1)
        return avg_loss, preds, targets

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² ignoring NaN targets."""
        mask = ~np.isnan(y_true)
        if mask.sum() < 2:
            return float("nan")
        ss_res = np.sum((y_true[mask] - y_pred[mask]) ** 2)
        ss_tot = np.sum((y_true[mask] - y_true[mask].mean()) ** 2)
        return 1 - ss_res / (ss_tot + 1e-12)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 50,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        train_loader = self._make_loader(train_df, shuffle=True,  fit_scaler=True)
        val_loader   = self._make_loader(val_df,   shuffle=False)

        best_val_loss = float("inf")
        print(f"Training on {self.device} | epochs={epochs}")

        for epoch in range(1, epochs + 1):
            tr_loss, _, _ = self._step(train_loader, train=True)
            val_loss, val_preds, val_targets = self._step(val_loader, train=False)

            # Per-target R²
            r2_scores = []
            for i, col in enumerate(TARGET_COLS):
                r2 = self._r2(val_targets[:, i], val_preds[:, i])
                r2_scores.append(r2)
            mean_r2 = np.nanmean(r2_scores)

            self.scheduler.step(val_loss)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_r2"].append(mean_r2)

            if epoch % 5 == 0 or epoch == 1:
                r2_str = " | ".join(
                    f"{col}:{r2:.3f}" for col, r2 in zip(TARGET_COLS, r2_scores)
                )
                print(
                    f"Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"R² [{r2_str}]"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(self.model.state_dict(), save_path)

        return self.history

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Return RMSE, MAE, R² per target on test set."""
        loader = self._make_loader(test_df, shuffle=False)
        _, preds, targets = self._step(loader, train=False)

        results = {}
        for i, col in enumerate(TARGET_COLS):
            mask = ~np.isnan(targets[:, i])
            if mask.sum() == 0:
                continue
            y_t = targets[mask, i]
            y_p = preds[mask, i]
            results[f"{col}_rmse"] = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
            results[f"{col}_mae"]  = float(np.mean(np.abs(y_t - y_p)))
            results[f"{col}_r2"]   = float(self._r2(y_t, y_p))

        return results


# ─────────────────────────────────────────────────────────
# Public predict() interface (dashboard-compatible)
# ─────────────────────────────────────────────────────────

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "reports" / "dl_model.pt"


def predict(
    df: pd.DataFrame,
    model: Optional[AquaSenseDLModel] = None,
    model_path: Optional[str | Path] = None,
    scaler=None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict chl_a, turbidity, do for a DataFrame of observations.

    Parameters
    ----------
    df         : DataFrame with at least the FEATURE_COLS columns.
    model      : pre-loaded AquaSenseDLModel. If None, loaded from model_path.
    model_path : path to saved model state-dict (.pt). Falls back to default.
    scaler     : feature scaler (must match training scaler).
    device     : 'cpu' or 'cuda'. Defaults to auto-detect.

    Returns
    -------
    pd.DataFrame with columns ['chl_a', 'turbidity', 'do'] and same index as df.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        path = Path(model_path or _DEFAULT_MODEL_PATH)
        model = AquaSenseDLModel()
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            warnings.warn(
                f"No model weights found at {path}. Returning untrained predictions.",
                UserWarning,
            )
    model = model.to(device)
    model.eval()

    if scaler is None:
        scaler_path = Path(__file__).parent.parent.parent / "reports" / "dl_scaler.pkl"
        if scaler_path.exists():
            try:
                import pickle
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            except Exception:
                pass

    # Build dataset (no targets needed for inference)
    df_inf = df.copy()
    for col in TARGET_COLS:
        df_inf[col] = np.nan    # placeholder so Dataset doesn't error

    ds = AquaSenseDataset(df_inf, scaler=scaler, fit_scaler=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            pred, _ = model(X_batch)
            all_preds.append(pred.cpu().numpy())

    preds = np.vstack(all_preds)
    return pd.DataFrame(preds, columns=TARGET_COLS, index=df.index)


# ─────────────────────────────────────────────────────────
# Smoke test (run: python -m src.models.dl_model)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AquaSense DL Model smoke test ===")

    # Build synthetic data matching Marutey's schema
    np.random.seed(42)
    N = 500
    synth = pd.DataFrame(
        {col: np.random.rand(N) for col in FEATURE_COLS}
    )
    synth["site"] = "Bellandur"
    synth["water_body_type"] = "lake"
    synth["lat"] = 12.93
    synth["lon"] = 77.64
    synth["date"] = pd.date_range("2024-01-01", periods=N, freq="D")
    synth["sensor"] = "S2"
    for tgt in TARGET_COLS:
        synth[tgt] = np.random.rand(N) * [50, 100, 8][TARGET_COLS.index(tgt)]

    train_df = synth.iloc[:400]
    val_df   = synth.iloc[400:]

    model   = AquaSenseDLModel()
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params:,}")

    trainer = AquaSenseTrainer(model, batch_size=32)
    # Quick 3-epoch train
    trainer.train(train_df, val_df, epochs=3)

    metrics = trainer.evaluate(val_df)
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test predict() interface
    out = predict(val_df, model=model)
    print(f"\npredict() output shape: {out.shape}")
    print(out.head())
    print("\n[OK] Smoke test passed.")
