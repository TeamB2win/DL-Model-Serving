import requests

from api.routes.inference import data_priority_queue
from model.model import DLModelHandler


dl_model = DLModelHandler()

def do_inference():
  while not data_priority_queue.empty():
    data = data_priority_queue.get()
    data = data[1]
    
    result = dl_model.inference(data) # 비디오 생성
    print(result)
    
    # Backend api request
    if data.prev_driving_path == "":
      response = fetch('post', result)
    else:
      response = fetch('put', result)
    print(response)
        
    if not result['isErr']:
      ...
      # if response.status == 400:
        # data_priority_queue.put((4, data))  # 큐에 가장 낮은 우선순위로 추가
    
def fetch(method, data):
  url = 'http://63.35.31.27:8000/dl'
  if method == 'post':
    response = requests.post(url, json=data)
  else:
    response = requests.put(url, json=data)
  return response