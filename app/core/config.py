from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Intelligent Lead Scoring System API"
    version: str = "0.1.0"
    model_path: str = "artifacts/lead_model.pkl"

    model_config = SettingsConfigDict(env_prefix="LEAD_API_", env_file=".env", extra="ignore")


settings = Settings()
