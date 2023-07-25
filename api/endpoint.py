from fastapi import APIRouter

from api.routes.inference import router as inference_router


router = APIRouter()

router.include_router(router=inference_router)