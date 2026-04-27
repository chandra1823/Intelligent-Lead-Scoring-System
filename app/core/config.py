from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Intelligent Lead Scoring System API"
    version: str = "0.1.0"
    # Used to derive artifact directory for hybrid model loading.
    model_path: str = "artifacts/random_forest.pkl"

    model_config = SettingsConfigDict(
        env_prefix="LEAD_API_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


settings = Settings()
