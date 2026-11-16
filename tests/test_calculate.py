import pytest

from tools import CalculateTool


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # Operator precedence and parentheses
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        # Division operators
        ("7 / 2", 3.5),
        ("7 // 2", 3),
        ("7 % 3", 1),
        # Unary operators: ** binds tighter than the unary minus
        ("-2 ** 2", -4),
        ("(-2) ** 2", 4),
        ("--5", 5),
        # Powers
        ("16 ** 0.5", 4.0),
        ("2 ** -1", 0.5),
        ("(-2) ** 3", -8),
        # Floats
        ("0.1 + 0.2", pytest.approx(0.3)),
        ("(24 * 60 - 935) / 60", pytest.approx(8.4167, abs=1e-4)),
    ],
)
def test_evaluates_valid_expressions(expression: str, expected: float) -> None:
    assert CalculateTool().run(expression=expression) == expected


def test_accepts_the_largest_allowed_integer() -> None:
    result = CalculateTool().run(expression="10 ** 999")

    assert result == 10**999


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        # Anything that is not arithmetic on numbers
        ("int(3.7)", "Unsupported expression"),
        ('__import__("os")', "Unsupported expression"),
        ("x + 1", "Unsupported expression"),
        ("True + 1", "Unsupported expression"),
        ('"a" * 3', "Unsupported expression"),
        ("2 < 3", "Unsupported expression"),
        ("[1, 2]", "Unsupported expression"),
        # Results too large to compute or send back (DoS protection)
        ("10 ** 1000", "Result too large"),
        ("9 ** 9 ** 9", "Result too large"),
        ("10 ** 600 * 10 ** 600", "Result too large"),
        ("1e308 * 10", "not a finite number"),
        # Complex results
        ("(-8) ** 0.5", "Negative base with a fractional exponent"),
    ],
)
def test_rejects_invalid_expressions(expression: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CalculateTool().run(expression=expression)


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        ("2 +", SyntaxError),
        ("1 / 0", ZeroDivisionError),
        ("0 ** -1", ZeroDivisionError),
    ],
)
def test_lets_other_errors_propagate(expression: str, error: type[Exception]) -> None:
    # The agent catches these and reports them to the model.
    with pytest.raises(error):
        CalculateTool().run(expression=expression)
