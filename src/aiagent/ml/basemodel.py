from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

class BaseModel(ABC):

    r"""
    It's a Base class for polimorphism at :meth:`registry.ModelRegistry`.

    BaseModel contains general methods for interacting with neural 
    network models.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

        self._config: Dict[str, Any] = {}
        self._model_path: Optional[Path] = None
        self._is_loaded = False

        self._device = "cpu"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    @property
    def model_path(self) -> str:
        return self._model_path
    @property
    def device(self) -> str:
        return self._device
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def load(self, model_path: Optional[str] = None, **kwargs) -> None:
        if self._is_loaded:
            print("Модель уже загружена")
            return
        
        model_path = model_path or self._config.get('model_path', '')
        if not model_path:
            raise ValueError("Не указан путь для 'model_path'!")
        
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError("Путь не существует!")
        
        self._load_impl(**kwargs)
        self._is_loaded = True
    
    def load_config(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._device = self._config.get('device', "cpu")

        path = self._config.get('model_path', "")
        if path:
            self._model_path = Path(path)

    @abstractmethod
    def _load_impl(self, **kwargs) -> None:
        ...
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        ...