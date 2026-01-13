from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "sentiment-service",
        "timestamp": datetime.utcnow().isoformat(),
    }

@router.get("/ready")
async def readiness_check():
    return {"status": "ready"}
