from queue import PriorityQueue

from schema.inference_schema import InferenceRequest 
 
  
class InferenceQueue(PriorityQueue):
  def __init__(self):
    super().__init__()
  
  def __contains__(self, item: InferenceRequest):
    for q in self.queue:
      if q[1] == item:
        return True
    return False