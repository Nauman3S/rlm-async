"""OpenRouter client wrapper"""

import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4.1-mini"


def get_api_key() -> str:
    """Get OpenRouter API key from environment"""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    return key


def get_client() -> OpenAI:
    """Get synchronous OpenAI client configured for OpenRouter"""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=get_api_key(),
    )


def get_async_client() -> AsyncOpenAI:
    """Get async OpenAI client configured for OpenRouter"""
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=get_api_key(),
    )
