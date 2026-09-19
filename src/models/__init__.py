"""src/models — heavy optional dependencies (xgboost/optuna, shap, torch) are imported lazily."""

from importlib import import_module

_LAZY = {
    "SpatialKFold": ".spatial_cv",
    "spatial_cross_val_score": ".spatial_cv",
    "MetricsReporter": ".metrics",
    "spatial_cv_rmse": ".metrics",
    "WaterQualityXGB": ".xgboost_pipeline",
    "SHAPExplainer": ".shap_explainer",
    "AquaSenseDLModel": ".dl_model",
    "AquaSenseTrainer": ".dl_model",
    "DLPredictor": ".dl_model",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
