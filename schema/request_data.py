from pydantic import BaseModel


class RequestData(BaseModel):
  id: int
  image_path: str
  wanted_type: bool
  prev_driving_path: str | None = ""
  video_path: str | None = ""
  
  def __eq__(self, other):
    if isinstance(other, RequestData):
      return self.image_path == other.image_path or self.id == other.id
  
    return False
  
  def set_video_path(self, path: str):
    self.video_path = path