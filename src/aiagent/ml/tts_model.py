from .basemodel import BaseModel

from typing import Any, Generator, Union, Optional

class TTSModel(BaseModel):

    def load(self, model_path: Optional[str] = None, **kwargs) -> None:
        r"""
        Load model this its dependences.

        Args:

            model_path (str, optional): 
                The path where the model is located.
            speaker_path (str, optional): 
                The path where the WAV-audio of the voice is located. Use for `one-shot`.
        """
        super().load(model_path, **kwargs)

    def _load_impl(self, **kwargs):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        config.load_json(f"{self._model_path}/config.json")

        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(
            config, 
            checkpoint_dir = self._model_path, 
            use_deepspeed = (self._device != "cpu")
        )
        self._model.to(self._device)

        speaker_wav_path = kwargs.get('speaker_path')
        if speaker_wav_path is not None:
            self.gpt_cond_latent, self.speaker_embedding = self._model.get_conditioning_latents(
                audio_path = speaker_wav_path
            )
        
    def generate(
        self, 
        text: str, 
        stream: bool = False,
        **kwargs
    ) -> Union[Generator[Any], Any]:
        r"""
        Get response from the model.

        Args:
            text (str): Text for voiceover.
            stream (bool, optional): 
                Set model generation `mode`. If it's `True`, model will be generate response
                'one-by-one', not all at once.
                
                Default `stream=False`
            **kwargs:
                Other options of generation.
                
                View :func:`TTS.tts.models.xtts.Xtts.inference_stream` documentation for more details.
        
        Returns:
            * A `wav-chunk` when `stream=False`;
            * A `generator of wav-chunk` when `stream=True`.
        """

        params = kwargs.copy()
        params['text'] = text
        if (
            hasattr(self, 'gpt_cond_latent') and
            hasattr(self, 'speaker_embedding')
        ):
            params.update(
                dict(
                    gpt_cond_latent = self.gpt_cond_latent, 
                    speaker_embedding = self.speaker_embedding
                )
            )

        if not stream:
            return self._model.inference(**params)['wav']

        for chunk in self._model.inference_stream(**params):
            yield chunk