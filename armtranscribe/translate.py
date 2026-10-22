"""Armenian -> English translation backends."""
from typing import Protocol


class Translator(Protocol):
    def translate(self, hy: str) -> str: ...
