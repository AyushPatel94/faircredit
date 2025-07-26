from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = ROOT_DIR / "data"
    raw_dir: Path = ROOT_DIR / "data" / "raw"
    windows_dir: Path = ROOT_DIR / "data" / "windows"
    logs_dir: Path = ROOT_DIR / "logs"

    mlflow_tracking_uri: str = f"sqlite:///{(ROOT_DIR / 'mlruns.db').as_posix()}"
    mlflow_artifact_root: str = f"file:{(ROOT_DIR / 'mlruns').as_posix()}"
    registered_model_name: str = "credit_risk"

    target_col: str = "TARGET"
    id_col: str = "SK_ID_CURR"
    protected_attributes: list[str] = ["CODE_GENDER", "AGE_BUCKET"]

    random_state: int = 42

    # production-simulation parameters
    n_weeks: int = 8
    rolling_window_weeks: int = 4
    drift_injection_start_week: int = 4
    drift_income_multiplier: float = 1.10

    # gate thresholds
    auc_epsilon: float = 0.005
    pr_auc_epsilon: float = 0.01
    brier_epsilon: float = 0.005
    auc_floor: float = 0.74
    # See ADR 002 for the 0.05 floor (Home Credit DP/EO ratios sit around
    # 0.10 in this dataset; 0.20 would block every promotion).
    dp_ratio_min: float = 0.05
    eo_ratio_min: float = 0.05
    # 2.0 instead of typical 0.25 because the warm-up weeks have very
    # different window sizes (week 0 trains on ~38k rows, week 3 on ~150k);
    # PSI between unequal samples is large by definition. Real production
    # with stable history can tighten this.
    drift_psi_max: float = 2.0
    audit_threshold: float = 0.15
    champion_max_age_days: int = 14

    log_level: str = "INFO"


settings = Settings()


XGB_DEFAULT_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}
