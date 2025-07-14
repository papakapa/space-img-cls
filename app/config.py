from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_PATH: str
    DEBUG: bool
    CORS_ORIGINS: list[str]
    API_RATE_LIMIT: str

    model_config = SettingsConfigDict(env_file=".env")

def settings_factory() -> Settings:
    return Settings()
