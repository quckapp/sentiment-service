from fastapi import APIRouter, Request
from typing import List

from app.schemas.sentiment import (
    AnalyzeTextRequest,
    AnalysisResult,
    BatchAnalyzeRequest,
    BatchAnalysisResult,
    ConversationSentimentRequest,
    ConversationSentimentResult,
    Sentiment,
)

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_text(request: AnalyzeTextRequest, req: Request):
    """Analyze sentiment and emotions of a single text"""
    service = req.app.state.sentiment_service
    return await service.analyze(
        request.text,
        include_emotions=request.include_emotions,
        include_keywords=request.include_keywords,
    )

@router.post("/analyze/batch", response_model=BatchAnalysisResult)
async def analyze_batch(request: BatchAnalyzeRequest, req: Request):
    """Analyze sentiment of multiple texts"""
    service = req.app.state.sentiment_service
    results = await service.analyze_batch(
        request.texts,
        include_emotions=request.include_emotions,
    )

    # Calculate distribution
    distribution = {"positive": 0, "negative": 0, "neutral": 0}
    total_score = 0

    for result in results:
        distribution[result.sentiment.sentiment.value] += 1
        if result.sentiment.sentiment == Sentiment.POSITIVE:
            total_score += result.sentiment.confidence
        elif result.sentiment.sentiment == Sentiment.NEGATIVE:
            total_score -= result.sentiment.confidence

    avg_score = total_score / len(results) if results else 0

    return BatchAnalysisResult(
        results=results,
        total=len(results),
        average_sentiment_score=round(avg_score, 3),
        sentiment_distribution=distribution,
    )

@router.post("/analyze/conversation", response_model=ConversationSentimentResult)
async def analyze_conversation(request: ConversationSentimentRequest, req: Request):
    """Analyze sentiment across a conversation"""
    service = req.app.state.sentiment_service
    result = await service.analyze_conversation(request.messages)
    return ConversationSentimentResult(**result)

@router.get("/models/info")
async def get_model_info(req: Request):
    """Get information about loaded models"""
    service = req.app.state.sentiment_service
    return {
        "sentiment_model": service.sentiment_classifier is not None,
        "emotion_model": service.emotion_classifier is not None,
        "status": "ready" if service.sentiment_classifier else "fallback",
    }
