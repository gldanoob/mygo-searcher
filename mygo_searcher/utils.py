from typing import get_args

def check_literal(literal: str, type) -> bool:
    values = get_args(type)
    return literal in values
