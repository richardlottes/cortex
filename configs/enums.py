from enum import Enum


class LLM(str, Enum):
    openai = "openai"
    anthropic = "anthropic"