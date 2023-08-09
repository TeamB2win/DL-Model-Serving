import time
import threading
from typing import Callable

from fastapi import FastAPI

from config.app import get_app_setting
from core.background_task import do_inference
from core.utils import clear_working_dir


running = True

class BackgroundTasks(threading.Thread):
  def run(self, *arg, **kwargs):
    global running
    settings = get_app_setting  
      
    while running:
      do_inference(settings)
    

def create_start_app_handler(
  app: FastAPI,
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
    global running
    
    running = False
    time.sleep(1)
    
    # clear_working_dir()
          
  return stop_app