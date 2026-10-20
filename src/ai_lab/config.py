from functools import cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    gemini_api_key: SecretStr
    default_model: str = "gemini-3.5-flash"
    max_tokens: int = 1024


@cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
