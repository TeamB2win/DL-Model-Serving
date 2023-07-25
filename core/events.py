from typing import Callable

from fastapi import FastAPI

from config.app import AppSetting


def create_start_app_handler(
  app: FastAPI,
  settings: AppSetting
) -> Callable:
  async def start_app() -> None:
    """GPU 자원을 확인 후 DL모델을 적재"""
    ...
    
  return start_app

def create_stop_app_handler(
  app: FastAPI
) -> Callable:
  async def stop_app() -> None:
    """DL 모델 GPU 메모리 상에서 삭제"""
    ...
    
  return stop_app