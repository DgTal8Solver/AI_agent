from .basemodel import BaseModel

from typing import Dict, Any, Generator, Union
from pydub import AudioSegment

class WhisperModel(BaseModel):

    _CHUNK_DURATION_MS = 1000   # 1 chunk - 1000 ms (1 second)

    def _load_impl(self, **kwargs):
        import whisper
        self._model = whisper.load_model(self._model_path, device = self._device)

    def generate(
        self, 
        audio, 
        **kwargs
    ) -> Union[Generator[Dict[str, Any]], Dict[str, Any]]:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded!")
        
        if not kwargs.get('stream', False):
            return self._model.transcribe(audio, **kwargs)
        
        audio = AudioSegment(audio)
        for i in range(0, len(audio), self._CHUNK_DURATION_MS):
            chunk = audio[i : i + self._CHUNK_DURATION_MS]
            if not chunk:
                continue
            result =  self._model.transcribe(chunk)
            yield result