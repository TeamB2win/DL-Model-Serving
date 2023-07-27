from queue import PriorityQueue
from schema.request_data import RequestData 
 
  
class InferenceQueue(PriorityQueue):
  def __init__(self):
    super().__init__()
  
  def __contains__(self, item: RequestData):
    for q in self.queue:
      if q[1] == item:
        return True
    return False