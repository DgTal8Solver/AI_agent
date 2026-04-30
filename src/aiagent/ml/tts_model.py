from .basemodel import BaseModel

class TTSModel(BaseModel):

    def _load_impl(self, **kwargs):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        
        pass