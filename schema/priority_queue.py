from pydantic import BaseModel
from queue import PriorityQueue


class RequestData(BaseModel):
  id: str
  type: int | None = 3
  image_path: str
  reinference: int | None = 0
  video_path: str | None = ""
  
  def __eq__(self, other):
    if isinstance(other, RequestData):
      return self.image_path == other.image_path
  
    return False
  
  def set_video_path(self, path: str):
    self.video_path = path
  
  
class InferenceQueue(PriorityQueue):
  def __init__(self):
    super().__init__()
  
  def __contains__(self, item: RequestData):
    for q in self.queue:
      if q[1] == item:
        return True
    return False