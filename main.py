from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.endpoint import router
from config.app import get_app_setting
from core.events import *


def get_application() -> FastAPI:
  # initialize FastAPI and settings
  application = FastAPI()
  settings = get_app_setting()
  
  # setting midware for cross-domain situation (react.js)
  application.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_host,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
  )
  
  # set event handler when the application is started
  application.add_event_handler(
    "startup",
    create_start_app_handler(application)    
  )
  
  # set event handler when the application shutdown
  application.add_event_handler(
    "shutdown", 
    create_stop_app_handler(application)
  )
  
  # inject FastAPI router
  application.include_router(router)
  
  return application

app = get_application()