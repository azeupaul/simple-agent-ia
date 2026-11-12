"""Tools the agent can ask the model to call."""

import ast
import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

from google.genai import types


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


class GetCurrentTimeTool(Tool):
    """Tool that returns the current time in ISO 8601 format."""

    name = "get_current_time"
    description = (
        "Get the current date and time in a given timezone. "
        "Use it whenever the user asks for the time or date somewhere, "
        "since you cannot know the current time otherwise."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. 'Europe/Paris'",
            }
        },
        "required": ["timezone"],
    }

    def run(self, **kwargs: Any) -> str:
        """Return the current time in ISO 8601 format.

        Args:
            **kwargs: Must contain ``timezone``, an IANA timezone name
            such as "Europe/Paris".

        Returns:
            The current date and time in ISO 8601 format, with the UTC offset.
        """
        timezone = kwargs["timezone"]

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
