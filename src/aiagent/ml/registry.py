from .basemodel import BaseModel
from typing import Type, Dict, Any

class ModelRegistry:

    _factories: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, factory: Type[BaseModel]) -> None:
        cls._factories[name] = factory
    
    @classmethod
    def remove(cls, name: str) -> None:
        if name in cls._factories.keys():
            cls._factories.pop(name)
    
    @classmethod
    def get(cls, name: str, config: Dict[str, Any]) -> Type[BaseModel]:
        factory = cls._factories.get(name, None)
        if factory is None:
            raise KeyError(f"Not registered model '{name}'!")
        
        model = factory(name)
        model.load_config(config)
        return model