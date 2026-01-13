import time
from typing import List, Dict, Optional
import structlog
from app.core.config import settings
from app.schemas.sentiment import (
    Sentiment,
    Emotion,
    SentimentResult,
    EmotionResult,
    AnalysisResult,
)

logger = structlog.get_logger()

class SentimentService:
    def __init__(self):
        self.sentiment_classifier = None
        self.emotion_classifier = None

    async def initialize(self):
        """Load ML models"""
        try:
            from transformers import pipeline

            logger.info("Loading sentiment model", model=settings.SENTIMENT_MODEL)
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model=settings.SENTIMENT_MODEL,
                top_k=None,
            )

            logger.info("Loading emotion model", model=settings.EMOTION_MODEL)
            self.emotion_classifier = pipeline(
                "text-classification",
                model=settings.EMOTION_MODEL,
                top_k=None,
            )

            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error("Failed to load models", error=str(e))
            # Fall back to TextBlob
            self.sentiment_classifier = None
            self.emotion_classifier = None

    async def analyze(
        self,
        text: str,
        include_emotions: bool = True,
        include_keywords: bool = False,
    ) -> AnalysisResult:
        """Analyze text for sentiment and emotions"""
        start_time = time.time()

        # Truncate long text
        text = text[:512]

        # Analyze sentiment
        sentiment_result = await self._analyze_sentiment(text)

        # Analyze emotions if requested
        emotion_result = None
        if include_emotions:
            emotion_result = await self._analyze_emotions(text)

        # Extract keywords if requested
        keywords = None
        if include_keywords:
            keywords = self._extract_keywords(text)

        processing_time = (time.time() - start_time) * 1000

        return AnalysisResult(
            text=text,
            sentiment=sentiment_result,
            emotions=emotion_result,
            keywords=keywords,
            word_count=len(text.split()),
            processing_time_ms=round(processing_time, 2),
        )

    async def _analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using transformer model"""
        if self.sentiment_classifier:
            try:
                results = self.sentiment_classifier(text)
                scores = {r["label"].lower(): r["score"] for r in results[0]}

                # Map to our sentiment enum
                if "positive" in scores:
                    sentiment = Sentiment.POSITIVE if scores.get("positive", 0) > 0.5 else (
                        Sentiment.NEGATIVE if scores.get("negative", 0) > 0.5 else Sentiment.NEUTRAL
                    )
                    confidence = max(scores.values())
                else:
                    # Handle different label formats
                    sentiment = Sentiment.NEUTRAL
                    confidence = 0.5

                return SentimentResult(
                    sentiment=sentiment,
                    confidence=confidence,
                    scores=scores,
                )
            except Exception as e:
                logger.error("Sentiment analysis failed", error=str(e))

        # Fallback to TextBlob
        return self._fallback_sentiment(text)

    async def _analyze_emotions(self, text: str) -> EmotionResult:
        """Analyze emotions using transformer model"""
        if self.emotion_classifier:
            try:
                results = self.emotion_classifier(text)
                emotions = {r["label"].lower(): r["score"] for r in results[0]}

                primary = max(emotions, key=emotions.get)
                confidence = emotions[primary]

                # Map to our emotion enum
                emotion_map = {
                    "joy": Emotion.JOY,
                    "happiness": Emotion.JOY,
                    "sadness": Emotion.SADNESS,
                    "anger": Emotion.ANGER,
                    "fear": Emotion.FEAR,
                    "surprise": Emotion.SURPRISE,
                    "disgust": Emotion.DISGUST,
                    "neutral": Emotion.NEUTRAL,
                }

                primary_emotion = emotion_map.get(primary, Emotion.NEUTRAL)

                return EmotionResult(
                    primary_emotion=primary_emotion,
                    confidence=confidence,
                    emotions=emotions,
                )
            except Exception as e:
                logger.error("Emotion analysis failed", error=str(e))

        # Fallback
        return EmotionResult(
            primary_emotion=Emotion.NEUTRAL,
            confidence=0.5,
            emotions={"neutral": 1.0},
        )

    def _fallback_sentiment(self, text: str) -> SentimentResult:
        """Fallback sentiment analysis using TextBlob"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                sentiment = Sentiment.POSITIVE
            elif polarity < -0.1:
                sentiment = Sentiment.NEGATIVE
            else:
                sentiment = Sentiment.NEUTRAL

            confidence = abs(polarity)

            return SentimentResult(
                sentiment=sentiment,
                confidence=min(confidence, 1.0),
                scores={
                    "positive": max(0, polarity),
                    "negative": max(0, -polarity),
                    "neutral": 1 - abs(polarity),
                },
            )
        except Exception:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.5,
                scores={"neutral": 1.0},
            )

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            # Get noun phrases as keywords
            keywords = list(blob.noun_phrases)[:max_keywords]
            return keywords
        except Exception:
            return []

    async def analyze_batch(
        self,
        texts: List[str],
        include_emotions: bool = True,
    ) -> List[AnalysisResult]:
        """Analyze multiple texts"""
        results = []
        for text in texts:
            result = await self.analyze(text, include_emotions)
            results.append(result)
        return results

    async def analyze_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict:
        """Analyze sentiment across a conversation"""
        results = []
        sentiment_by_user = {}

        for msg in messages:
            text = msg.get("text", "")
            user_id = msg.get("user_id", "unknown")

            result = await self.analyze(text, include_emotions=False)
            results.append(result)

            if user_id not in sentiment_by_user:
                sentiment_by_user[user_id] = {"positive": 0, "negative": 0, "neutral": 0}

            sentiment_by_user[user_id][result.sentiment.sentiment.value] += 1

        # Calculate overall sentiment
        positive = sum(1 for r in results if r.sentiment.sentiment == Sentiment.POSITIVE)
        negative = sum(1 for r in results if r.sentiment.sentiment == Sentiment.NEGATIVE)
        neutral = len(results) - positive - negative

        if positive > negative:
            overall = Sentiment.POSITIVE
        elif negative > positive:
            overall = Sentiment.NEGATIVE
        else:
            overall = Sentiment.NEUTRAL

        # Determine trend (simplified)
        if len(results) >= 3:
            recent = results[-3:]
            recent_positive = sum(1 for r in recent if r.sentiment.sentiment == Sentiment.POSITIVE)
            if recent_positive >= 2:
                trend = "improving"
            elif recent_positive == 0:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "overall_sentiment": overall,
            "sentiment_trend": trend,
            "message_count": len(messages),
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "sentiment_by_user": sentiment_by_user,
        }
