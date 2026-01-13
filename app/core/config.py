from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "sentiment-service"
    PORT: int = 5017
    DEBUG: bool = False

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Models
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"

    # Cache
    CACHE_TTL: int = 3600  # 1 hour

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
