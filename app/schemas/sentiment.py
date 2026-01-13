from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Emotion(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    include_emotions: bool = True
    include_keywords: bool = False

class SentimentResult(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(..., ge=0, le=1)
    scores: Dict[str, float]

class EmotionResult(BaseModel):
    primary_emotion: Emotion
    confidence: float = Field(..., ge=0, le=1)
    emotions: Dict[str, float]

class AnalysisResult(BaseModel):
    text: str
    sentiment: SentimentResult
    emotions: Optional[EmotionResult] = None
    keywords: Optional[List[str]] = None
    word_count: int
    processing_time_ms: float

class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    include_emotions: bool = True

class BatchAnalysisResult(BaseModel):
    results: List[AnalysisResult]
    total: int
    average_sentiment_score: float
    sentiment_distribution: Dict[str, int]

class ConversationSentimentRequest(BaseModel):
    messages: List[Dict[str, str]]  # [{user_id, text, timestamp}]
    workspace_id: Optional[str] = None
    channel_id: Optional[str] = None

class ConversationSentimentResult(BaseModel):
    overall_sentiment: Sentiment
    sentiment_trend: str  # improving, declining, stable
    message_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    sentiment_by_user: Dict[str, Dict[str, float]]
