"""Tools the agent can ask the model to call."""

import ast
import inspect
import math
import operator
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

from google.genai import types

# Python types allowed for tool parameters, mapped to their JSON Schema type.
_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_SECTION_HEADERS = {"Args:", "Returns:", "Raises:", "Yields:", "Examples:", "Note:"}


def _is_header(line: str) -> bool:
    """Return True if the line starts a docstring section, such as "Args:"."""
    return line.strip() in _SECTION_HEADERS


def _description(lines: list[str]) -> str:
    """Return the text before the first section header."""
    description = []
    for line in lines:
        if _is_header(line):
            break
        description.append(line)
    return "\n".join(description).strip()


def _section(lines: list[str], header: str) -> list[str]:
    """Return the lines of a section, without its header."""
    section: list[str] = []
    inside = False
    for line in lines:
        if _is_header(line):
            inside = line.strip() == header
            continue
        if inside:
            section.append(line)
    return section


def _parse_args(lines: list[str]) -> dict[str, str]:
    """Map each parameter name to its description in an Args section."""
    args: dict[str, str] = {}
    current = None
    for line in lines:
        if line.startswith("        "):  # 8 spaces: continuation line
            if current is not None:
                args[current] += " " + line.strip()
        elif line.startswith("    "):  # 4 spaces: new parameter
            name, _, text = line.strip().partition(":")
            current = name
            args[name] = text.strip()
    return args


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a description and parameter docs."""
    lines = doc.splitlines()
    return _description(lines), _parse_args(_section(lines, "Args:"))


def build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build the JSON Schema of a function's parameters.

    Each parameter becomes a property: its JSON type comes from its type
    annotation, and its description from the Args section of the function's
    Google-style docstring. Parameters without a default value are required.

    Args:
        func: The function to describe. Every parameter must be annotated with
            one of the types in ``_JSON_TYPES`` and documented in ``Args:``.

    Returns:
        A JSON Schema object describing the function's parameters.

    Raises:
        TypeError: If the function uses ``*args`` or ``**kwargs``, if a
            parameter has no type annotation, an unsupported type or no
            description, or if the docstring documents a parameter that the
            function does not have.
    """
    _, param_docs = parse_docstring(inspect.getdoc(func) or "")
    signature = inspect.signature(func)
    hints = typing.get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        hint = hints.get(name)
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(f"{func.__name__} cannot use *args or **kwargs as a tool")

        if hint is None:
            raise TypeError(
                f"Parameter {name!r} of {func.__name__} has no type annotation"
            )

        if hint not in _JSON_TYPES:
            raise TypeError(
                f"Parameter '{name}' of {func.__name__} has "
                f"unsupported type {getattr(hint, '__name__', hint)}"
            )

        description = param_docs.get(name, "")
        if not description:
            raise TypeError(
                f"Parameter {name!r} of {func.__name__} has no description "
                "in the Args section of its docstring"
            )

        properties[name] = {"type": _JSON_TYPES[hint], "description": description}

        if param.default is inspect.Parameter.empty:
            required.append(name)

    unknown = set(param_docs) - signature.parameters.keys()
    if unknown:
        raise TypeError(
            f"The docstring of {func.__name__} documents unknown parameters: "
            f"{sorted(unknown)}"
        )

    return {"type": "object", "properties": properties, "required": required}


class Tool(ABC):
    """Base class for a tool the model can ask the agent to run.

    Subclasses must define the class attributes below and implement `run`.

    Attributes:
        name: Unique identifier the model uses to request the tool.
        description: What the tool does. The model reads it to decide when
            to request the tool, so it should be clear and specific.
        parameters: JSON Schema of the arguments. The model reads it to
            decide which arguments to send.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check that subclasses define the required class attributes."""
        super().__init_subclass__(**kwargs)
        for attr in ("name", "description", "parameters"):
            if not getattr(cls, attr, None):
                raise TypeError(f"{cls.__name__} must define a non-empty '{attr}'")

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Run the tool with the arguments requested by the model.

        Args:
            **kwargs: Arguments matching the `parameters` schema.

        Returns:
            The result sent back to the model. It must be JSON-serializable.

        Raises:
            Exception: If the tool fails. The agent will catch it and send
                an error message back to the model.
        """

    @property
    def declaration(self) -> types.FunctionDeclaration:
        """The tool's declaration, sent to the model so it knows the tool exists."""
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.parameters,
        )


def tool(func: Callable[..., Any]) -> Tool:
    """Turn a documented function into a tool the model can call.

    The tool's name is the function's name, its description is the docstring's
    description, and its parameters schema is built from the signature and
    the ``Args:`` section of the docstring.

    Args:
        func: The function to wrap. It must have a Google-style docstring and
            a type annotation on every parameter.

    Returns:
        A tool instance that calls ``func`` with the arguments sent by the model.

    Raises:
        TypeError: If the docstring has no description, or if the parameters
            cannot be described (see ``build_parameters_schema``).
    """
    func_description, _ = parse_docstring(inspect.getdoc(func) or "")

    class FunctionTool(Tool):
        name = func.__name__
        description = func_description
        parameters: ClassVar[dict[str, Any]] = build_parameters_schema(func)

        def run(self, **kwargs: Any) -> Any:
            return func(**kwargs)

    return FunctionTool()


@tool
def get_current_time(timezone: str) -> str:
    """Get the current date and time in a given timezone.

    Use it whenever the user asks for the time or date somewhere, since you
    cannot know the current time otherwise.

    Args:
        timezone: IANA timezone name, e.g. "Europe/Paris". Use the timezone
            of the main city if the user gives a country.

    Returns:
        The current date and time in ISO 8601 format, with the UTC offset.
    """
    return datetime.now(ZoneInfo(timezone)).isoformat()


class CalculateTool(Tool):
    """Tool that evaluates a mathematical expression."""

    name = "calculate"
    description = (
        "Evaluate an arithmetic expression and return the exact result. "
        "Always use it instead of calculating in your head, especially with "
        "large or decimal numbers. "
        "Supported: numbers, + - * / // % ** and parentheses. "
        "Not supported: functions (sqrt, round...), variables, dates or times. "
        "Use x ** 0.5 for a square root, and convert times to minutes or "
        "seconds before calculating."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "Arithmetic expression in Python syntax, e.g. "
                    "'(24 * 60 - 935) / 60'. Use ** for powers, not ^."
                ),
            }
        },
        "required": ["expression"],
    }

    _MAX_DIGITS: ClassVar[int] = 1000  # arbitrary limit to prevent DoS

    _BINARY_OPERATORS: ClassVar[
        dict[type[ast.operator], Callable[[float, float], float]]
    ] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _UNARY_OPERATORS: ClassVar[dict[type[ast.unaryop], Callable[[float], float]]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def run(self, **kwargs: Any) -> float:
        """Evaluate a mathematical expression and return the result.

        Args:
            **kwargs: Must contain ``expression``, a mathematical expression
                such as "2 + 2 * 3".

        Returns:
            The result of the mathematical expression.
        """
        tree = ast.parse(kwargs["expression"], mode="eval")

        result = self._evaluate(tree.body)

        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Result is not a finite number")

        if isinstance(result, int) and abs(result) >= 10**self._MAX_DIGITS:
            raise ValueError("Result too large")

        return result

    def _check_power(self, base: float, exponent: float) -> None:
        """Check that a power can be computed safely, before computing it.

        Args:
            base: The base of the power operation.
            exponent: The exponent of the power operation.

        Raises:
            ValueError: If the base is negative and the exponent fractional
                (the result would be a complex number), or if the result
                would have more than ``_MAX_DIGITS`` digits.
        """
        if base < 0 and not (isinstance(exponent, int) or exponent.is_integer()):
            raise ValueError("Negative base with a fractional exponent")

        # The result has about |exponent| * |log10(|base|)| digits. Comparing
        # |exponent| to _MAX_DIGITS / |log10(|base|)| avoids converting a huge
        # integer exponent to float, which would overflow.
        digits_per_unit = abs(math.log10(abs(base))) if base != 0 else 0.0
        if digits_per_unit and abs(exponent) > self._MAX_DIGITS / digits_per_unit:
            raise ValueError("Result too large")

    def _evaluate(self, node: ast.expr) -> float:
        """Recursively evaluate an AST node representing a mathematical expression."""
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return node.value

        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UNARY_OPERATORS:
            unary_op = self._UNARY_OPERATORS[type(node.op)]
            return unary_op(self._evaluate(node.operand))

        if isinstance(node, ast.BinOp) and type(node.op) in self._BINARY_OPERATORS:
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            if isinstance(node.op, ast.Pow):
                self._check_power(left, right)

            binary_op = self._BINARY_OPERATORS[type(node.op)]
            return binary_op(left, right)

        raise ValueError(f"Unsupported expression: {ast.unparse(node)!r}")


class ToolRegistry:
    """A collection of tools, indexed by name."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        """Initialize the registry with the given tools.

        Args:
            tools: The list of tools to register.
        """
        self._tools: dict[str, Tool] = {}

        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tools to register.

        Raises:
            ValueError: If a tool with same name already exists.
        """
        if tool.name in self._tools:
            raise ValueError(f"A tool named {tool.name!r} is already registered")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool with this name, or None if there is none."""
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        """The names of the registered tools."""
        return list(self._tools)

    @property
    def declarations(self) -> list[types.FunctionDeclaration]:
        """The declarations of the registered tools."""
        return [item.declaration for item in self._tools.values()]

    def __len__(self) -> int:
        """Count all registered tools.

        Returns:
            The number of tools in the registry.
        """
        return len(self._tools)
