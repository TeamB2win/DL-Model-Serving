from fastapi import APIRouter

from api.routes.test import router as test_router


router = APIRouter()

router.include_router(router=test_router)
