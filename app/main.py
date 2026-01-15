from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from app.api import sentiment, health
from app.core.config import settings
from app.services.sentiment_service import SentimentService

logger = structlog.get_logger()

sentiment_service: SentimentService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sentiment_service
    logger.info("Starting sentiment service", port=settings.PORT)
    sentiment_service = SentimentService()
    await sentiment_service.initialize()
    app.state.sentiment_service = sentiment_service
    yield
    logger.info("Shutting down sentiment service")

app = FastAPI(
    title="Sentiment Service",
    description="QuckApp Sentiment Analysis and Emotion Detection Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["Sentiment"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
