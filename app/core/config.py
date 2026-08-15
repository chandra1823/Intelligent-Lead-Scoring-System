from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Intelligent Lead Scoring System API"
    version: str = "2.0.0"

    # ------------------------------------------------------------- storage
    database_url: str = f"sqlite:///{ROOT / 'data' / 'leadscoring.db'}"
    sql_echo: bool = False
    artifact_root: str = str(ROOT / "artifacts")

    # ------------------------------------------------------------ security
    # Encrypts connector credentials at rest. Generate a real one for any
    # deployment: python -c "import secrets;print(secrets.token_urlsafe(32))"
    secret_key: str = "dev-only-insecure-key-change-me"
    # When false, unauthenticated requests act as the default tenant. Useful
    # for a single-user local install; must be true anywhere else.
    require_api_key: bool = False
    default_tenant_slug: str = "default"

    allowed_origins: list[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ]

    # --------------------------------------------------------- rate limiting
    rate_limit_per_minute: int = 240
    rate_limit_enabled: bool = True

    # -------------------------------------------------------------- scoring
    decision_threshold: float = 0.5
    # Half-life in days for score decay on stale leads.
    score_decay_half_life_days: float = 21.0
    score_decay_enabled: bool = True

    # ------------------------------------------------------------- training
    tier1_min_labels: int = 1
    tier2_min_labels: int = 500
    tier3_min_labels: int = 5000
    # A challenger must beat the champion by this much AUC to be promoted.
    promotion_margin: float = 0.005

    # ----------------------------------------------------------- monitoring
    drift_warn_psi: float = 0.1
    drift_alert_psi: float = 0.25

    # Legacy field retained so existing .env files keep working.
    model_path: str = "artifacts/base/random_forest.pkl"

    model_config = SettingsConfigDict(
        env_prefix="LEAD_API_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    @property
    def base_artifact_dir(self) -> Path:
        return Path(self.artifact_root) / "base"

    def tenant_artifact_dir(self, tenant_id: str, version: str | None = None) -> Path:
        base = Path(self.artifact_root) / "tenants" / tenant_id
        return base / version if version else base


settings = Settings()
