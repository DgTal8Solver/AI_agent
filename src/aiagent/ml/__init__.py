from .whisper_model import WhisperModel
from .tts_model import TTSModel
from .registry import ModelRegistry

ModelRegistry.register("whisper", WhisperModel)
ModelRegistry.register("tts", TTSModel)

__all__ = ["ModelRegistry"]