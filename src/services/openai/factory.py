"""Factory for creating OpenAI client instances."""

from src.config import get_settings
from .client import OpenAIClient


def make_openai_client() -> OpenAIClient:
    """Create OpenAI client instance."""
    settings = get_settings()
    return OpenAIClient(settings)
