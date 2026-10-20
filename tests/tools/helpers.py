import pytest

from quantagent.errors import QuantError


def err(fn, **kwargs) -> QuantError:
    with pytest.raises(QuantError) as e:
        fn(**kwargs)
    return e.value


def has(warnings: list[str], text: str) -> bool:
    return any(text.lower() in w.lower() for w in warnings)
