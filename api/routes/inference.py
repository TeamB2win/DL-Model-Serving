import os
import time

from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse
from starlette.status import HTTP_200_OK

from model.model import DLModelHandler
from schema.priority_queue import InferenceQueue, RequestData


dl_model = DLModelHandler()
data_priority_queue = InferenceQueue()

router = APIRouter(prefix="/api", tags=["api"])

@router.post("/inference")
async def add_data(data: RequestData):
  if data in data_priority_queue: # 추론 큐에 데이터가 존재하는 지 확인
    print("already exsist data in Inference Queue")
  else:
    # TODO: image path 유효성 검사
    
    # Enqueue the data into the priority queue with the calculated priority
    data_priority_queue.put((data.type, data))  # 큐에 데이터 추가
    print("success")
    
  return HTMLResponse(
    content="OK",
    status_code=HTTP_200_OK
  )
  
def do_inference():
  while not data_priority_queue.empty():
    data = data_priority_queue.get()
    data = data[1]
    
    result = dl_model.inference(data) # 비디오 생성
    
    # 데이터 처리 결과를 필요한 방식으로 사용하거나 저장
    print("Inference result:", result)
    
    # TODO Backend api request
    # TODO 응답을 받아서 다시 추론하거나 등등 처리
    
    # sync with aiohttp.ClientSession() as session:
    #   url = "http://63.35.31.27:20325/api/inference_result"
    #   async with session.post(url, json=result_data) as response:
    #       print(await response.text())
    print("Backend server request")    