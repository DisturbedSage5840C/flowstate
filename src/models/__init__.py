"""src/models/__init__.py"""
from .spatial_cv import SpatialKFold, spatial_cross_val_score
from .xgboost_pipeline import WaterQualityXGB
from .metrics import MetricsReporter, spatial_cv_rmse
from .shap_explainer import SHAPExplainer

try:
    from .dl_model import AquaSenseDLModel, AquaSenseTrainer
except ImportError:
    pass

__all__ = [
    "SpatialKFold",
    "spatial_cross_val_score",
    "WaterQualityXGB",
    "MetricsReporter",
    "spatial_cv_rmse",
    "SHAPExplainer",
]
