from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSetting(BaseSettings):
  backend_url: str
  video_dir: str
  working_dir: str
  allowed_host: list[str] = ["*"]
    
  model_config = SettingsConfigDict(env_file='.env')


@lru_cache
def get_app_setting() -> AppSetting:
  return AppSetting()