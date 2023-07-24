from typing import Callable

from fastapi import FastAPI

from config.app import AppSetting


def create_start_app_handler(
  app: FastAPI,
  settings: AppSetting
) -> Callable:
  async def start_app() -> None:
    ...
    
  return start_app

def create_stop_app_handler(
  app: FastAPI
) -> Callable:
  async def stop_app() -> None:
    ...
    
  return stop_app