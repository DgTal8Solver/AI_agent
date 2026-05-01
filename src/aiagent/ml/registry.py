from .basemodel import BaseModel
from typing import Type, Dict, Any

class ModelRegistry:
    r'''
    `ModelRegistry` allows to load registered models dependences only they are initialized,
    instead of loading heavy libraries at once.
    
    .. ## Before:

        # Import modules with heavy libraries: whisper, torch, transformers and etc.
        import whisper_model as wm
        import other_model as om

        # Using
        whisper_model = wm.WhisperModel(...)
        other_model = om.OtherModel(...)

        whisper_model.generate(...)
        other_model(...)
        
    .. ## After:

        # Import only ModelRegistry
        from registry import ModelRegistry

        # Using
        whisper_model = ModelRegistry.get('whisper') # Load only 'whisper' package
        other_model = ModelRegistry.get('other') # Load only 'torch' package

        whisper_model.generate(...)
        other_model(...)
    
    '''

    _factories: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, factory: Type[BaseModel]) -> None:
        cls._factories[name] = factory
    
    @classmethod
    def remove(cls, name: str) -> None:
        if name not in cls._factories.keys():
            raise ValueError("Incorrect model name! Can't find it!")
        cls._factories.pop(name)
        
    
    @classmethod
    def get(cls, name: str, config: Dict[str, Any]) -> Type[BaseModel]:
        factory = cls._factories.get(name, None)
        if factory is None:
            raise KeyError(f"Not registered model '{name}'!")
        
        model = factory(name)
        model.load_config(config)
        return model