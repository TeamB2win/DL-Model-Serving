from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from starlette.status import HTTP_200_OK

from models.priority_queue import InferenceQueue, RequestData 


QUEUE = InferenceQueue()
    
router = APIRouter(prefix="/test", tags=["test"])

@router.post(
  '/task',
  tags=['task']
)
async def add_data_to_pq(data: RequestData):
  """추론 요청 큐에 데이터 추가

  Args:
      data (RequestData): 
        id: 수배 아이디
        type: 긴급 / 종합 여부 긴급: 1, 종합: 3
        image_path: 공유 폴더 내에 저장된 이미지 주소
        is_reinference: 재추론 요청일 시 우선순위 2

  Returns:
      HTMLResponse: 성공 메시지
  """
  
  if data in QUEUE: # 추론 큐에 데이터가 존재하는 지 확인
    print("already exsist data in Inference Queue")
  else:
    # TODO: image path 유효성 검사
    
    QUEUE.put((data.type, data))  # 큐에 데이터 추가
    print("success")
    
  return HTMLResponse(
    content="OK",
    status_code=HTTP_200_OK
  )