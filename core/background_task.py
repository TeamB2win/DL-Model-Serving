import requests, os

from config.app import AppSetting
from api.routes.inference import data_priority_queue
from model.model import DLModelHandler
from core.utils import clear_working_dir


dl_model = DLModelHandler()

def do_inference(settings: AppSetting):
  if not data_priority_queue.empty():
    print(f"========= Getting start inference =========")    
    print(f"current items in inference queue: {data_priority_queue.qsize()}")
    
    data = data_priority_queue.get()
    data = data[1]
    print(f"get data in inference queue: {data}") 
       
    result = dl_model.inference(data) # 비디오 생성
    print(f"result: {result}")
    
    if result['is_err']:
      clear_working_dir()
      
    # 이전에 생성된 비디오 삭제
    if data.video_path and os.path.exists(data.video_path):
      os.remove(data.video_path)
      print("Delete previous video")
    
    # Backend api request
    if data.prev_driving_path is None:
      method = 'post'
    else:
      method = 'put'
      
    url = os.path.join(settings.backend_url, 'dl')
    try:
      fetch(url, method, result)
      print("backend request succeed")
    except:
      print("error")
      result = {
        'id': data.id,
        'is_err': True,
        'err_msg': "Fail to result request to backend have to inference again"
      }
      requests.post(url, result)

    
def fetch(url: str, method: str, data: dict):
  if method == 'post':
    response = requests.post(url, json=data)
  else:
    response = requests.put(url, json=data)
  return response