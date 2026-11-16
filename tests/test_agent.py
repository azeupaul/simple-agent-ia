from typing import Any, cast, override

import pytest
from google import genai
from google.genai import errors, types

from agent import Agent, AgentObserver
from tools import tool


class FakeModels:
    """Stands in for ``client.models``: replays scripted responses, records calls."""

    def __init__(self) -> None:
        self.responses: list[types.GenerateContentResponse | Exception] = []
        self.calls: list[list[types.Content]] = []

    def generate_content(
        self, *, model: str, contents: list[types.Content], config: Any
    ) -> types.GenerateContentResponse:
        self.calls.append(list(contents))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    """Stands in for ``genai.Client`` so that tests never call Gemini."""

    def __init__(self) -> None:
        self.models = FakeModels()


def reply(*parts: types.Part) -> types.GenerateContentResponse:
    """Build a Gemini response whose single candidate contains these parts."""
    content = types.Content(role="model", parts=list(parts))
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def text(value: str) -> types.Part:
    return types.Part.from_text(text=value)


def call(name: str, **args: Any) -> types.Part:
    return types.Part.from_function_call(name=name, args=args)


def function_responses(content: types.Content) -> list[dict[str, Any] | None]:
    """Return the responses of the function response parts of a message."""
    return [
        part.function_response.response
        for part in content.parts or []
        if part.function_response
    ]


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first integer.
        b: The second integer.
    """
    return a + b


@tool
def fail() -> str:
    """Always fail."""
    raise ValueError("boom")


class RecordingObserver(AgentObserver):
    """Records the agent's events as short strings, so tests can compare them."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.durations: list[float] = []

    @override
    def on_model_request(self, iteration: int) -> None:
        self.events.append(f"model #{iteration}")

    @override
    def on_tool_call(self, call: types.FunctionCall) -> None:
        self.events.append(f"call {call.name}")

    @override
    def on_tool_result(
        self, name: str, response: dict[str, Any], duration: float
    ) -> None:
        # The response has a single key: "result" or "error".
        [kind] = response
        self.events.append(f"{kind} {name}")
        self.durations.append(duration)

    @override
    def on_final_answer(self, iterations: int) -> None:
        self.events.append(f"answer after {iterations}")


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def observer() -> RecordingObserver:
    return RecordingObserver()


@pytest.fixture
def agent(client: FakeClient, observer: RecordingObserver) -> Agent:
    return Agent(
        cast(genai.Client, client),
        model="fake-model",
        tools=[add, fail],
        observer=observer,
    )


# Plain conversation


def test_returns_the_model_text(client: FakeClient, agent: Agent) -> None:
    client.models.responses = [reply(text("Hello!"))]

    assert agent.chat("Hi") == "Hello!"


def test_remembers_previous_turns(client: FakeClient, agent: Agent) -> None:
    client.models.responses = [reply(text("Nice to meet you")), reply(text("Paul"))]

    agent.chat("My name is Paul")
    agent.chat("What is my name?")

    second_request = client.models.calls[1]
    assert [content.role for content in second_request] == ["user", "model", "user"]
    assert len(agent.history) == 4


def test_returns_an_empty_string_when_the_reply_has_no_text(
    client: FakeClient, agent: Agent
) -> None:
    client.models.responses = [reply()]

    assert agent.chat("Hi") == ""


def test_reset_clears_the_history(client: FakeClient, agent: Agent) -> None:
    client.models.responses = [reply(text("Hello!"))]
    agent.chat("Hi")

    agent.reset()

    assert agent.history == ()


# Tool calls


def test_runs_a_tool_and_sends_the_result_back(
    client: FakeClient, agent: Agent
) -> None:
    client.models.responses = [reply(call("add", a=2, b=3)), reply(text("It is 5."))]

    assert agent.chat("What is 2 + 3?") == "It is 5."

    tool_message = client.models.calls[1][-1]
    assert tool_message.role == "user"
    assert function_responses(tool_message) == [{"result": 5}]


def test_runs_parallel_tool_calls_in_one_message(
    client: FakeClient, agent: Agent
) -> None:
    client.models.responses = [
        reply(call("add", a=1, b=1), call("add", a=2, b=2)),
        reply(text("2 and 4.")),
    ]

    agent.chat("What are 1 + 1 and 2 + 2?")

    tool_message = client.models.calls[1][-1]
    assert function_responses(tool_message) == [{"result": 2}, {"result": 4}]


def test_reports_an_unknown_tool_to_the_model(client: FakeClient, agent: Agent) -> None:
    client.models.responses = [reply(call("weather")), reply(text("Sorry."))]

    agent.chat("What is the weather?")

    [response] = function_responses(client.models.calls[1][-1])
    assert response == {
        "error": "Unknown tool 'weather'. Available tools: ['add', 'fail']"
    }


def test_reports_a_tool_error_to_the_model(client: FakeClient, agent: Agent) -> None:
    client.models.responses = [reply(call("fail")), reply(text("It failed."))]

    assert agent.chat("Try it") == "It failed."

    [response] = function_responses(client.models.calls[1][-1])
    assert response == {"error": "Tool request failed: ValueError('boom')"}


def test_stores_the_whole_turn_in_the_history(client: FakeClient, agent: Agent) -> None:
    client.models.responses = [reply(call("add", a=2, b=3)), reply(text("It is 5."))]

    agent.chat("What is 2 + 3?")

    # user message, tool call, tool result, final answer
    assert [content.role for content in agent.history] == [
        "user",
        "model",
        "user",
        "model",
    ]


# Limits and failures


def test_stops_after_max_iterations(client: FakeClient) -> None:
    agent = Agent(
        cast(genai.Client, client), model="fake-model", tools=[add], max_iterations=2
    )
    client.models.responses = [reply(call("add", a=1, b=1)) for _ in range(3)]

    with pytest.raises(RuntimeError, match="kept requesting tools"):
        agent.chat("Loop forever")

    assert len(client.models.calls) == 2
    assert agent.history == ()


@pytest.mark.parametrize("max_iterations", [0, -1])
def test_rejects_less_than_one_iteration(
    client: FakeClient, max_iterations: int
) -> None:
    with pytest.raises(ValueError, match="max_iterations must be at least 1"):
        Agent(
            cast(genai.Client, client),
            model="fake-model",
            max_iterations=max_iterations,
        )


@pytest.mark.parametrize(
    "error",
    [
        errors.APIError(503, {"error": {"message": "Unavailable"}}),
        TimeoutError("network down"),
    ],
    ids=["api-error", "network-error"],
)
def test_wraps_request_failures_in_runtime_error(
    client: FakeClient, agent: Agent, error: Exception
) -> None:
    client.models.responses = [error]

    with pytest.raises(RuntimeError, match="Gemini request failed"):
        agent.chat("Hi")


def test_rejects_a_response_without_candidates(
    client: FakeClient, agent: Agent
) -> None:
    client.models.responses = [types.GenerateContentResponse(candidates=[])]

    with pytest.raises(RuntimeError, match="no candidates"):
        agent.chat("Hi")


def test_keeps_the_history_unchanged_when_a_turn_fails(
    client: FakeClient, agent: Agent
) -> None:
    client.models.responses = [
        reply(text("Hello!")),
        reply(call("add", a=1, b=1)),
        TimeoutError("network down"),
    ]
    agent.chat("Hi")
    history_before = agent.history

    with pytest.raises(RuntimeError):
        agent.chat("What is 1 + 1?")

    assert agent.history == history_before


# Observer


def test_notifies_every_step_of_a_turn_in_order(
    client: FakeClient, agent: Agent, observer: RecordingObserver
) -> None:
    client.models.responses = [reply(call("add", a=2, b=3)), reply(text("It is 5."))]

    agent.chat("What is 2 + 3?")

    assert observer.events == [
        "model #1",
        "call add",
        "result add",
        "model #2",
        "answer after 2",
    ]


def test_notifies_failed_and_unknown_tools(
    client: FakeClient, agent: Agent, observer: RecordingObserver
) -> None:
    client.models.responses = [
        reply(call("fail"), call("weather")),
        reply(text("Nothing worked.")),
    ]

    agent.chat("Try them")

    assert observer.events == [
        "model #1",
        "call fail",
        "error fail",
        "call weather",
        "error weather",
        "model #2",
        "answer after 2",
    ]


def test_measures_a_positive_tool_duration(
    client: FakeClient, agent: Agent, observer: RecordingObserver
) -> None:
    client.models.responses = [reply(call("add", a=2, b=3)), reply(text("It is 5."))]

    agent.chat("What is 2 + 3?")

    [duration] = observer.durations
    assert duration >= 0


def test_does_not_announce_an_answer_when_the_limit_is_reached(
    client: FakeClient, observer: RecordingObserver
) -> None:
    agent = Agent(
        cast(genai.Client, client),
        model="fake-model",
        tools=[add],
        max_iterations=1,
        observer=observer,
    )
    client.models.responses = [reply(call("add", a=1, b=1))]

    with pytest.raises(RuntimeError):
        agent.chat("Loop forever")

    assert observer.events == ["model #1", "call add", "result add"]


def test_works_without_an_observer(client: FakeClient) -> None:
    agent = Agent(cast(genai.Client, client), model="fake-model", tools=[add])
    client.models.responses = [reply(call("add", a=2, b=3)), reply(text("It is 5."))]

    assert agent.chat("What is 2 + 3?") == "It is 5."
