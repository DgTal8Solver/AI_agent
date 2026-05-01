from .basemodel import BaseModel

from typing import Dict, Any, Generator, Union, Optional
from pydub import AudioSegment


class WhisperModel(BaseModel):

    _CHUNK_DURATION_MS = 1000   # 1 chunk - 1000 ms (1 second)

    def load(self, model_path: Optional[str] = None, **kwargs) -> None:
        r"""
        Load model this its dependences.

        Args:

            model_path (str, optional): 
                The path where the model is located.
            in_memory (bool, optional): 
                Whether to preload the model weights into host memory.
                
                View :func:`whisper.load_model` documentation for more details.
        """
        super().load(model_path, **kwargs)

    def _load_impl(self, **kwargs):
        import whisper
        self._model = whisper.load_model(self._model_path, device = self._device, **kwargs)

    def generate(
        self, 
        audio, 
        stream: bool = False,
        **kwargs
    ) -> Union[Generator[Dict[str, Any]], Dict[str, Any]]:
        r"""
        Get response from the model.

        Args:
            audio (Any): Audio array.
            stream (bool, optional): 
                Set model generation `mode`. If it's `True`, model will be generate response
                'one-by-one', not all at once.
                
                Default `stream=False`
            **kwargs:
                Other options of generation.
                
                View :func:`whisper.Whisper.transcribe` documentation for more details.
        
        Returns:
            * A `dictionary` when `stream=False`;
            * A `generator of dictionaries` when `stream=True`.

            View :func:`whisper.Whisper.transcribe` documentation, title `Returns` for more details.
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded!")
        
        if not stream:
            return self._model.transcribe(audio, **kwargs)
        
        # TODO: Find more effective and correct method of output streaming
        audio = AudioSegment(audio)
        for i in range(0, len(audio), self._CHUNK_DURATION_MS):
            chunk = audio[i : i + self._CHUNK_DURATION_MS]
            if not chunk:
                continue
            result =  self._model.transcribe(chunk, **kwargs)
            yield result