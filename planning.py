"""Planning: a request split into steps, then run one step at a time."""

from pydantic import BaseModel, Field

MAX_STEPS = 5


class Step(BaseModel):
    """One step of a plan."""

    description: str = Field(
        min_length=1,
        description="One concrete action, in the user's language, that can be "
        "done with the available tools or by reasoning and writing.",
    )


class Plan(BaseModel):
    """A request split into a few ordered steps."""

    goal: str = Field(
        min_length=1,
        description="The user's request, restated in one sentence.",
    )
    steps: list[Step] = Field(
        min_length=1,
        max_length=MAX_STEPS,
        description=f"The steps to reach the goal, in order, {MAX_STEPS} at most. "
        "Do not add steps that are not needed.",
    )

    def lines(self) -> list[str]:
        """Return the steps, numbered, one per line."""
        return [
            f"{index}. {step.description}"
            for index, step in enumerate(self.steps, start=1)
        ]

    def step_prompt(self, index: int) -> str:
        """Return the message that asks the model to do one step (index from 1).

        The whole plan is repeated, so the model knows where the step fits,
        even if earlier turns were summarized in the meantime.
        """
        plan = "\n".join(self.lines())
        return (
            f"We are following this plan to reach the goal: {self.goal}\n"
            f"{plan}\n\n"
            f"Do step {index} now, and only this step: "
            f"{self.steps[index - 1].description}"
        )

    def final_prompt(self) -> str:
        """Return the message that asks for the final answer, once all steps ran."""
        return (
            "All the steps of the plan are done. Using their results, give the "
            f"final answer to the goal: {self.goal}"
        )
