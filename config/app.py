from functools import lru_cache


class AppSetting():
  allowed_host: list[str] = ["*"]


@lru_cache
def get_app_setting() -> AppSetting:
  return AppSetting()