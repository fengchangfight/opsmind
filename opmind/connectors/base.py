from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Awaitable
from opmind.models import Document


class BaseConnector(ABC):
    connector_name: str
    supported_types: list[str] = []

    @abstractmethod
    async def extract(self, source: str) -> AsyncIterator[Document]:
        ...

    async def validate(self, source: str) -> bool:
        return True

    async def watch(
        self,
        source: str,
        callback: Callable[[Document], Awaitable[None]],
    ) -> None:
        raise NotImplementedError("CDC not implemented")
