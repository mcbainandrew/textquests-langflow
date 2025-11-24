"""
Reflection System for Self-Evolving Society of Mind

Analyzes gameplay logs to identify failure patterns and suggest improvements.
"""

from .failure_detector import FailureDetector, FailureReport
from .llm_analyzer import LLMAnalyzer, ImprovementSuggestion

__all__ = [
    "FailureDetector",
    "FailureReport",
    "LLMAnalyzer",
    "ImprovementSuggestion",
]
