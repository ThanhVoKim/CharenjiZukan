from abc import ABC, abstractmethod

class BaseTranslationProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def call(self, message: str) -> str:
        ...
