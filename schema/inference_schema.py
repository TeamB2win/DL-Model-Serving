from pydantic import Field

from schema.base import BaseSchemaModel


class InferenceRequest(BaseSchemaModel):
  id: int
  image_path: str
  wanted_type: bool
  prev_driving_path: str | None = ""
  video_path: str | None = ""
  
  def __eq__(self, other):
    if isinstance(other, InferenceRequest):
      return self.id == other.id
    return False
  
  def __lt__(self, other):
    if isinstance(other, InferenceRequest):
      return self.id < other.id
    return False
  
  def __le__(self, other):
    if isinstance(other, InferenceRequest):
      return self.id <= other.id
    return False
  
  def set_video_path(self, path: str):
    self.video_path = path
    

class InferenceResponse(BaseSchemaModel):
  message: str = Field(default="succeed")
  status: str = Field(default="OK")
  