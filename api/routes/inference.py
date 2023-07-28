import os
import cv2
import imageio

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from starlette.status import HTTP_200_OK, HTTP_208_ALREADY_REPORTED, HTTP_500_INTERNAL_SERVER_ERROR

from schema.priority_queue import InferenceQueue, RequestData


data_priority_queue = InferenceQueue()

router = APIRouter(prefix="/api", tags=["api"])

@router.post("/inference")
async def add_data(data: RequestData):
  # 추론 큐에 데이터가 존재하는 지 확인
  if data in data_priority_queue: 
    print("already exsist data in Inference Queue")
    return HTMLResponse(
      content="Already",
      status_code=HTTP_208_ALREADY_REPORTED
    )

  # image path 유효성 검사
  if not os.path.exists(data.image_path):
    return HTMLResponse(
      content="Invalid Image Path",
      status_code=HTTP_500_INTERNAL_SERVER_ERROR
    )
    
  # image file을 열 수 있는지 검사 
  try:
    imageio.imread(data.image_path)
    cv2.imread(data.image_path)
  except:
    return HTMLResponse(
      content="Invalid Image Path",
      status_code=HTTP_500_INTERNAL_SERVER_ERROR
    )
    
  # Enqueue the data into the priority queue with the calculated priority
  data_priority_queue.put((data.type, data))  # 큐에 데이터 추가
    
  return HTMLResponse(
    content="OK",
    status_code=HTTP_200_OK
  )
