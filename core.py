from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

from joblib import dump, load

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor  # v2.1.x

# --------------------------- Config helpers ---------------------------

def set_pandas_transform_output() -> None:
    """
    Make sklearn transformers emit pandas DataFrames (keeps feature names).
    """
    set_config(transform_output="pandas")

# --------------------------- Feature prep -----------------------------

def ensure_features(
    df: pd.DataFrame,
    final_features: List[str],
    target: str,
    time_col: str = "timeClose"
) -> pd.DataFrame:
    """
    Idempotent feature preparation:
      - parse & sort time
      - add high_ret_1, log_volume, log_marketCap if missing
      - convert to numeric and light forward/back fill within features
    """
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.sort_values(time_col).reset_index(drop=True)

    if "high" in d.columns and "high_ret_1" in final_features and "high_ret_1" not in d.columns:
        d["high_ret_1"] = d["high"].pct_change()

    if "marketCap" in d.columns and "log_marketCap" in final_features and "log_marketCap" not in d.columns:
        d["log_marketCap"] = np.log1p(pd.to_numeric(d["marketCap"], errors="coerce"))

    if "volume" in d.columns and "log_volume" in final_features and "log_volume" not in d.columns:
        d["log_volume"] = np.log1p(pd.to_numeric(d["volume"], errors="coerce"))

    # unify dtype and light fill
    for col in final_features:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d[final_features] = d[final_features].ffill().bfill()

    # sanity checks
    miss = [c for c in final_features if c not in d.columns]
    if miss:
        raise ValueError(f"Missing required features: {miss}")
    if target not in d.columns:
        raise ValueError(f"Missing target column: {target}")

    return d

# --------------------------- Split -----------------------------------

def fixed_chrono_split(
    df: pd.DataFrame,
    final_features: List[str],
    target: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Returns: (x_train, y_train, x_val, y_val, x_test, y_test, X_all_ordered)
    """
    m = df.dropna(subset=[target]).reset_index(drop=True)
    n = len(m)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)

    X = m[final_features]
    y = m[target].to_numpy()

    x_train, y_train = X.iloc[:n_tr], y[:n_tr]
    x_val,   y_val   = X.iloc[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
    x_test,  y_test  = X.iloc[n_tr+n_va:],     y[n_tr+n_va:]

    return x_train, y_train, x_val, y_val, x_test, y_test, X

# --------------------------- Metrics ---------------------------------

def mape(y, yhat) -> float:  return float(np.mean(np.abs((y - yhat)/(y + 1e-12))))
def smape(y, yhat) -> float: return float(np.mean(2*np.abs(y - yhat)/(np.abs(y)+np.abs(yhat)+1e-12)))

def report(tag: str, y_true, y_pred) -> str:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    msg = (f"{tag:>5} | MAE={mae:.6f}  RMSE={rmse:.6f}  "
           f"MAPE={mape(y_true, y_pred):.4f}  sMAPE={smape(y_true, y_pred):.4f}  R2={r2:.6f}")
    print(msg)
    return msg

# --------------------------- Models ----------------------------------

def build_rf(
    n_estimators: int = 600,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    n_jobs: int = -1,
    with_scaler: bool = False
) -> Pipeline:
    """
    RandomForest in a Pipeline with SimpleImputer (and optional StandardScaler).
    """
    steps = [("imp", SimpleImputer(strategy="median"))]
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("rf", RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        random_state=random_state, n_jobs=n_jobs
    )))
    return Pipeline(steps)

def build_xgb(
    n_estimators: int = 2000,
    learning_rate: float = 0.02,
    max_depth: int = 7,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 2.0,
    early_stopping_rounds: int = 100,
    random_state: int = 42,
    n_jobs: int = -1
) -> Pipeline:
    """
    XGBRegressor (v2.1+) in a Pipeline with SimpleImputer; eval args set in constructor.
    """
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("xgb", xgb)])

# --------------------------- Train/Eval ------------------------------

def fit_with_optional_val(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None
) -> Pipeline:
    """
    Fits a Pipeline. If the final step is XGB, passes eval_set for early stopping.
    """
    step_name, est = model.steps[-1]
    if isinstance(est, XGBRegressor) and x_val is not None and y_val is not None:
        model.fit(x_train, y_train, **{f"{step_name}__eval_set": [(x_val, y_val)]})
    else:
        model.fit(x_train, y_train)
    return model

def _metrics_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

def eval_on_splits(
    tag: str,
    model: Pipeline,
    x_train: pd.DataFrame, y_train: np.ndarray,
    x_val: pd.DataFrame,   y_val: np.ndarray,
    x_test: pd.DataFrame,  y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Prints and returns metrics on train/val/test.
    """
    out = {}
    print(f"\n=== {tag} ===")
    p_tr = model.predict(x_train); out["Train"] = _metrics_dict(y_train, p_tr); report("Train", y_train, p_tr)
    p_va = model.predict(x_val);   out["Val"]   = _metrics_dict(y_val,   p_va); report(" Val ", y_val,   p_va)
    p_te = model.predict(x_test);  out["Test"]  = _metrics_dict(y_test,  p_te); report("Test", y_test,  p_te)
    return out

# --------------------------- Predict/Artifacts -----------------------

def predict_last(model: Pipeline, X_all: pd.DataFrame, features: List[str]) -> float:
    """
    Predict next-day HIGH for the last row of X_all[features].
    """
    return float(model.predict(X_all[features].iloc[[-1]])[0])

def save_artifact(
    path: str | Path,
    model: Pipeline,
    features: List[str],
    target: str,
    token: str = "XRP",
    mode: str = "level"
) -> str:
    """
    Save model + metadata (feature order, target, token, mode).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "features": features, "target": target, "token": token, "mode": mode}, str(path))
    return str(path)

def load_artifact(path: str | Path) -> Dict[str, Any]:
    """
    Load joblib artifact dictionary.
    """
    return load(str(path))
