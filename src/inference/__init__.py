"""Real-time inference and spoken Urdu synthesis engine.

Maintains a 30-frame rolling keypoint buffer, evaluates ST-GCN predictions
in real-time at >= 30 FPS, and plays pre-cached Urdu pronunciation audio.
"""

from src.inference.audio_engine import AudioEngine
from src.inference.live_demo import RealTimePSLTranslator, run_live_webcam
from src.inference.translation_layer import (
    BilingualTranslator,
    LanguageMode,
    SentenceStyle,
    TranslatedOutput,
)

__all__ = [
    "AudioEngine",
    "RealTimePSLTranslator",
    "run_live_webcam",
    "BilingualTranslator",
    "LanguageMode",
    "SentenceStyle",
    "TranslatedOutput",
]
