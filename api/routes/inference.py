import os
import cv2
import imageio

from fastapi import APIRouter, status

from api.errors.http_errors import HTTP_Exception
from schema.priority_queue import InferenceQueue
from schema.inference_schema import *
from resource.strings import DESCRIPTION_400_ERROR_FOR_DATA_FILE


data_priority_queue = InferenceQueue()

ImagePathException = HTTP_Exception(
  status_code=status.HTTP_400_BAD_REQUEST,
  description=DESCRIPTION_400_ERROR_FOR_DATA_FILE,
  detail="Can't access the image path"
)
ImageOpenException = HTTP_Exception(
    status_code=status.HTTP_400_BAD_REQUEST,
    description=DESCRIPTION_400_ERROR_FOR_DATA_FILE,
    detail="Can't open image by using image framework(imageio or cv2)"
)

router = APIRouter(prefix="/api", tags=["api"])

@router.post(
  path="/inference",
  response_model=InferenceResponse,
  responses={
    **ImagePathException.responses,
    **ImageOpenException.responses
  },
  name="inference: insert data into inference queue"
)
async def add_data(data: InferenceRequest) -> InferenceResponse:
  # 추론 큐에 데이터가 존재하는 지 확인
  if data in data_priority_queue: 
    print("already exist in the inference queue")
    return InferenceResponse(
      message="already exist in the inference queue",
      status="OK"
    )

  # image path 유효성 검사
  if not os.path.exists(data.image_path):
    ImagePathException.error_raise()
    
  # image file을 열 수 있는지 검사 
  try:
    imageio.imread(data.image_path)
    cv2.imread(data.image_path)
  except:
    ImageOpenException.error_raise()
    
  # Enqueue the data into the priority queue with the calculated priority
  data_priority_queue.put((data.wanted_type - 1, data))  # 큐에 데이터 추가
  
  return InferenceResponse()
