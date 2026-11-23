import pytest
from pydantic import ValidationError

from tasks import Task, TaskList


def test_a_task_line_shows_its_priority_title_and_deadline() -> None:
    task = Task(title="Send the report", priority="high", due="Friday")

    assert task.line() == "[high]   Send the report (Friday)"


def test_a_task_line_without_deadline() -> None:
    assert Task(title="Call the plumber", priority="low").line() == (
        "[low]    Call the plumber"
    )


def test_lines_put_the_most_urgent_tasks_first() -> None:
    task_list = TaskList(
        tasks=[
            Task(title="a", priority="low"),
            Task(title="b", priority="high"),
            Task(title="c", priority="medium"),
            Task(title="d", priority="high"),
        ]
    )

    # Tasks of the same priority keep the order in which they were mentioned.
    assert [line.split()[-1] for line in task_list.lines()] == ["b", "d", "c", "a"]


def test_parses_the_json_of_the_model() -> None:
    task_list = TaskList.model_validate_json(
        '{"tasks": [{"title": "Send the report", "priority": "high", "due": null}]}'
    )

    assert task_list.tasks == [Task(title="Send the report", priority="high")]


@pytest.mark.parametrize(
    "task",
    [
        {"title": "", "priority": "high"},
        {"title": "Send the report", "priority": "urgent"},
        {"priority": "high"},
    ],
    ids=["empty-title", "unknown-priority", "no-title"],
)
def test_rejects_an_invalid_task(task: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        TaskList.model_validate({"tasks": [task]})


def test_the_schema_describes_the_fields_to_the_model() -> None:
    schema = TaskList.model_json_schema()
    task = schema["$defs"]["Task"]

    assert task["properties"]["priority"]["enum"] == ["low", "medium", "high"]
    assert task["required"] == ["title", "priority"]
    assert all("description" in field for field in task["properties"].values())
    assert "INSTRUCTION" not in schema["properties"]  # a ClassVar is not a field
