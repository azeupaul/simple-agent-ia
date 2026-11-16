import pytest

from tools import ToolRegistry, tool


@tool
def first() -> str:
    """Return the first value."""
    return "first"


@tool
def second() -> str:
    """Return the second value."""
    return "second"


def test_an_empty_registry() -> None:
    registry = ToolRegistry()

    assert len(registry) == 0
    assert not registry
    assert registry.names == []
    assert registry.declarations == []


def test_registers_the_tools_given_at_creation() -> None:
    registry = ToolRegistry([first, second])

    assert len(registry) == 2
    assert registry.names == ["first", "second"]
    assert [d.name for d in registry.declarations] == ["first", "second"]


def test_get_returns_the_tool_with_this_name() -> None:
    registry = ToolRegistry([first, second])

    assert registry.get("second") is second


def test_get_returns_none_for_an_unknown_name() -> None:
    assert ToolRegistry([first]).get("unknown") is None


def test_register_adds_a_tool() -> None:
    registry = ToolRegistry([first])

    registry.register(second)

    assert registry.names == ["first", "second"]


def test_rejects_two_tools_with_the_same_name() -> None:
    registry = ToolRegistry([first])

    with pytest.raises(ValueError, match="'first' is already registered"):
        registry.register(first)

    assert len(registry) == 1
