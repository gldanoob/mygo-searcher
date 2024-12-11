from typing import Literal

from utils import check_literal

Methods = Literal['embedding', 'literal']


class SemanticSearcher():
    def __init__(self, method: Methods = 'embedding') -> None:
        if not check_literal(method, Methods):
            raise ValueError(f"Invalid method. Must be one of {Methods}")
        self.method = method

    def add_dataset(self, dataset: list[str]) -> None:
        pass

    def search(self, query: str, top_k=5) -> list[str]:
        pass
