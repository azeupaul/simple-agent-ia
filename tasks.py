"""Structured output: the tasks the user mentioned in the conversation.

The field descriptions end up in the JSON Schema sent to the model: like a
tool's description, they are a prompt.
"""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

Priority = Literal["low", "medium", "high"]


class Task(BaseModel):
    """A thing the user said they have to do."""

    title: str = Field(
        min_length=1,
        description="What to do, as a short sentence in the user's language.",
    )
    priority: Priority = Field(
        description="How urgent or important the user made it sound. "
        "Use medium when the user gave no clue."
    )
    due: str | None = Field(
        default=None,
        description="The deadline as the user said it, e.g. 'Friday' or "
        "'before noon'. Null if the user gave none.",
    )

    def line(self) -> str:
        """Return the task on one line, e.g. "[high]   Send the report (Friday)"."""
        due = f" ({self.due})" if self.due else ""
        return f"{f'[{self.priority}]':<8} {self.title}{due}"


class TaskList(BaseModel):
    """The tasks mentioned in a conversation."""

    tasks: list[Task] = Field(
        description="Every task the user mentioned, in the order they were "
        "mentioned. Empty if there is none."
    )

    # A ClassVar is not a field: Pydantic leaves it out of the schema.
    INSTRUCTION: ClassVar[str] = (
        "List every task the user said they have to do in this conversation, "
        "including the ones already mentioned in a summary."
    )

    def lines(self) -> list[str]:
        """Return the tasks, most urgent first, one per line."""
        order = {"high": 0, "medium": 1, "low": 2}
        ranked = sorted(self.tasks, key=lambda task: order[task.priority])
        return [task.line() for task in ranked]
