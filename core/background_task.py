import requests, os

from api.routes.inference import data_priority_queue
from model.model import DLModelHandler


dl_model = DLModelHandler()

def do_inference():
  while not data_priority_queue.empty():
    data = data_priority_queue.get()
    data = data[1]
    
    result = dl_model.inference(data) # 비디오 생성
    
    # 이전에 생성된 비디오 삭제
    if data.video_path and os.path.exists(data.video_path):
      os.remove(data.video_path)
      print("Delete previous video")
    
    print(result)
    
    # Backend api request
    if data.prev_driving_path == "":
      response = fetch('post', result)
    else:
      response = fetch('put', result)

    # # TODO 예외처리 백엔드랑 소통 필요  
    # if hasattr(response, "status_code"):
    #   ...
      # if response.status_code == '400':
      #   print("insert data into priority queue again")
      #   data_priority_queue.put((4, data))  # 큐에 가장 낮은 우선순위로 추가
    
def fetch(method, data):
  url = 'http://63.35.31.27:8000/dl'
  if method == 'post':
    response = requests.post(url, json=data)
  else:
    response = requests.put(url, json=data)
  return response