"""src/models/__init__.py"""
from .spatial_cv import SpatialKFold, spatial_cross_val_score
from .metrics import MetricsReporter, spatial_cv_rmse

__all__ = [
    "SpatialKFold",
    "spatial_cross_val_score",
    "MetricsReporter",
    "spatial_cv_rmse",
]

# WaterQualityXGB (needs optuna), SHAPExplainer (needs shap), and the DL model
# (needs torch) each pull in a heavy optional dependency. Importing them
# eagerly meant `from src.models.dl_model import ...` failed with a missing
# optuna/shap even when only the DL stack was needed. Import lazily instead.
try:
    from .xgboost_pipeline import WaterQualityXGB
    __all__.append("WaterQualityXGB")
except ImportError:
    pass

try:
    from .shap_explainer import SHAPExplainer
    __all__.append("SHAPExplainer")
except ImportError:
    pass

try:
    from .dl_model import AquaSenseDLModel, AquaSenseTrainer
    __all__ += ["AquaSenseDLModel", "AquaSenseTrainer"]
except ImportError:
    pass
