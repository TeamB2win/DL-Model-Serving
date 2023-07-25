import time
import threading
from typing import Callable

from fastapi import FastAPI

from config.app import AppSetting
from api.routes.test import QUEUE


class BackgroundTasks(threading.Thread):
  def run(self, *arg, **kwargs):
    while True:
      while not QUEUE.empty():
        q = QUEUE.get()
        
        # TODO: 추론 코드
        
        time.sleep(10)
        print(q)
        print("task_run")
        print()  
          
        # TODO: Backend로 비디오주소 포함 request
    # Thread 종료 조건
    

def create_start_app_handler(
  app: FastAPI,
  settings: AppSetting
) -> Callable:
  async def start_app() -> None:
    # TODO: GPU 자원을 확인 후 DL모델을 적재
    
    t = BackgroundTasks()
    t.start()
    
  return start_app


def create_stop_app_handler(
  app: FastAPI
) -> Callable:
  async def stop_app() -> None:
    # TODO: DL 모델 GPU 메모리 상에서 삭제
    ...
    
  return stop_app