import pytest
from pydantic import ValidationError

from planning import MAX_STEPS, Plan, Step


@pytest.fixture
def plan() -> Plan:
    return Plan(
        goal="Know how long is left before midnight in Tokyo",
        steps=[
            Step(description="Get the current time in Tokyo"),
            Step(description="Compute the time left before midnight"),
        ],
    )


def test_lines_number_the_steps(plan: Plan) -> None:
    assert plan.lines() == [
        "1. Get the current time in Tokyo",
        "2. Compute the time left before midnight",
    ]


def test_step_prompt_repeats_the_plan_and_names_the_step(plan: Plan) -> None:
    prompt = plan.step_prompt(2)

    assert "goal: Know how long is left before midnight in Tokyo" in prompt
    assert "1. Get the current time in Tokyo" in prompt  # the whole plan
    assert prompt.endswith(
        "Do step 2 now, and only this step: Compute the time left before midnight"
    )


def test_final_prompt_asks_for_the_goal(plan: Plan) -> None:
    assert plan.final_prompt().endswith(
        "final answer to the goal: Know how long is left before midnight in Tokyo"
    )


@pytest.mark.parametrize(
    "steps",
    [[], [{"description": "Step"}] * (MAX_STEPS + 1), [{"description": ""}]],
    ids=["no-step", "too-many-steps", "empty-step"],
)
def test_rejects_an_invalid_plan(steps: list[dict[str, str]]) -> None:
    with pytest.raises(ValidationError):
        Plan.model_validate({"goal": "Do something", "steps": steps})


def test_the_schema_limits_the_number_of_steps() -> None:
    steps = Plan.model_json_schema()["properties"]["steps"]

    assert steps["minItems"] == 1
    assert steps["maxItems"] == MAX_STEPS
