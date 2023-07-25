from pydantic import BaseModel
from queue import PriorityQueue


class InferenceQueue(PriorityQueue):
  def __init__(self):
    super().__init__()
  
  def __contains__(self, item):
    for q in self.queue:
      if q[1] == item:
        return True
    return False
  
  
class RequestData(BaseModel):
  id: str
  type: int | None = 3
  image_path: str
  is_reinference: bool | None = False
  
  def __eq__(self, other):
    if isinstance(other, RequestData):
      return self.image_path == other.image_path
  
    return False