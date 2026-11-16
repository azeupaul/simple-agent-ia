import inspect
from collections.abc import Callable
from typing import Any

import pytest

from tools import Tool, build_parameters_schema, parse_docstring, tool


def get_weather(city: str, days: int = 1, metric: bool = True) -> str:
    """Get the weather forecast for a city.

    Use it whenever the user asks about the weather.

    Args:
        city: Name of the city, e.g. "Paris". Use the main city if the user
            gives a country.
        days: Number of days to forecast.
        metric: Whether to use Celsius instead of Fahrenheit.

    Returns:
        A short forecast.
    """
    return f"{city}: sunny for {days} day(s), metric={metric}"


def _getdoc(func: Callable[..., Any]) -> str:
    """Return the docstring without its indentation, as the tools module reads it."""
    return inspect.getdoc(func) or ""


# parse_docstring


def test_parse_docstring_keeps_every_paragraph_of_the_description() -> None:
    description, _ = parse_docstring(_getdoc(get_weather))

    assert description == (
        "Get the weather forecast for a city.\n\n"
        "Use it whenever the user asks about the weather."
    )


def test_parse_docstring_joins_continuation_lines() -> None:
    _, args = parse_docstring(_getdoc(get_weather))

    assert args["city"] == (
        'Name of the city, e.g. "Paris". Use the main city if the user gives a country.'
    )


def test_parse_docstring_reads_only_the_args_section() -> None:
    _, args = parse_docstring(_getdoc(get_weather))

    assert list(args) == ["city", "days", "metric"]


def test_parse_docstring_without_args_section() -> None:
    assert parse_docstring("Do something.") == ("Do something.", {})


def test_parse_docstring_of_an_empty_docstring() -> None:
    assert parse_docstring("") == ("", {})


# build_parameters_schema


def test_build_parameters_schema() -> None:
    assert build_parameters_schema(get_weather) == {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    'Name of the city, e.g. "Paris". Use the main city if the '
                    "user gives a country."
                ),
            },
            "days": {"type": "integer", "description": "Number of days to forecast."},
            "metric": {
                "type": "boolean",
                "description": "Whether to use Celsius instead of Fahrenheit.",
            },
        },
        "required": ["city"],
    }


def test_build_parameters_schema_of_a_function_without_parameters() -> None:
    def ping() -> str:
        """Check that the service is up."""
        return "pong"

    assert build_parameters_schema(ping) == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_build_parameters_schema_rejects_var_args() -> None:
    def func(*values: int) -> int:
        """Add numbers.

        Args:
            values: The numbers to add.
        """
        return sum(values)

    with pytest.raises(TypeError, match=r"cannot use \*args or \*\*kwargs"):
        build_parameters_schema(func)


def test_build_parameters_schema_rejects_var_kwargs() -> None:
    def func(**options: str) -> str:
        """Show options.

        Args:
            options: The options to show.
        """
        return str(options)

    with pytest.raises(TypeError, match=r"cannot use \*args or \*\*kwargs"):
        build_parameters_schema(func)


def test_build_parameters_schema_rejects_missing_annotation() -> None:
    def func(city):  # type: ignore[no-untyped-def]
        """Do something.

        Args:
            city: A city.
        """

    with pytest.raises(TypeError, match="'city' of func has no type annotation"):
        build_parameters_schema(func)


def test_build_parameters_schema_rejects_unsupported_type() -> None:
    def func(cities: list[str]) -> None:
        """Do something.

        Args:
            cities: Some cities.
        """

    with pytest.raises(TypeError, match="'cities' of func has unsupported type"):
        build_parameters_schema(func)


@pytest.mark.parametrize(
    "docstring",
    [
        "Do something.",
        "Do something.\n\nArgs:\n    other: Another parameter.",
        "Do something.\n\nArgs:\n    city:",
    ],
    ids=["no-args-section", "not-documented", "empty-description"],
)
def test_build_parameters_schema_rejects_undocumented_parameter(
    docstring: str,
) -> None:
    def func(city: str) -> None:
        pass

    func.__doc__ = docstring

    with pytest.raises(TypeError, match="'city' of func has no description"):
        build_parameters_schema(func)


def test_build_parameters_schema_rejects_unknown_documented_parameter() -> None:
    def func(city: str) -> None:
        """Do something.

        Args:
            city: A city.
            country: A country.
        """

    with pytest.raises(TypeError, match=r"documents unknown parameters: \['country'\]"):
        build_parameters_schema(func)


# @tool


def test_tool_builds_a_tool_from_the_function() -> None:
    weather = tool(get_weather)

    assert isinstance(weather, Tool)
    assert weather.name == "get_weather"
    assert weather.description == parse_docstring(_getdoc(get_weather))[0]
    assert weather.parameters == build_parameters_schema(get_weather)


def test_tool_run_calls_the_function_with_the_model_arguments() -> None:
    calls: list[dict[str, Any]] = []

    @tool
    def record(city: str, days: int = 1) -> str:
        """Record a call.

        Args:
            city: A city.
            days: A number of days.
        """
        calls.append({"city": city, "days": days})
        return "recorded"

    assert record.run(city="Paris", days=3) == "recorded"
    assert calls == [{"city": "Paris", "days": 3}]


def test_tool_declaration_uses_the_generated_schema() -> None:
    declaration = tool(get_weather).declaration

    assert declaration.name == "get_weather"
    assert declaration.parameters_json_schema == build_parameters_schema(get_weather)


def test_tool_rejects_a_function_without_description() -> None:
    def ping() -> str:
        return "pong"

    with pytest.raises(TypeError, match="must define a non-empty 'description'"):
        tool(ping)
