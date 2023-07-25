import os
import time

from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse
from starlette.status import HTTP_200_OK

from schema.priority_queue import InferenceQueue, RequestData


# 추론 더비 함수
def make_video(data: RequestData):
  time.sleep(10)
  print(data)
  data.set_video_path("/test")
  
  return data


data_priority_queue = InferenceQueue()

router = APIRouter(prefix="/api", tags=["api"])

@router.post("/inference")
async def sort_data(data: RequestData):
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
    result_data = []
    data = data_priority_queue.get()
    data = data[1]
    # TODO: data 전처리 함수
    
    
    # TODO: 비디오 추론함수
    data = make_video(data)
    print(data)
                
    ### 추론이 끝났으므로 request_data.req_again TURE로 변경
    result_data.append(data)
    
    # 데이터 처리 결과를 필요한 방식으로 사용하거나 저장
    print("Inference result:", data)
    
    print("Backend server request")    
    # sync with aiohttp.ClientSession() as session:
    #   url = "http://63.35.31.27:20325/api/inference_result"
    #   async with session.post(url, json=result_data) as response:
    #       print(await response.text())  

    # await asyncio.sleep(0.1)