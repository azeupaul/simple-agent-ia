"""Conversational agent backed by a Gemini model."""

from collections.abc import Sequence

from google import genai
from google.genai import types

from tools import Tool


class Agent:
    """A conversational agent that keeps the history of its exchanges with Gemini.

    Each call to `chat` sends the whole conversation to the model, so the agent
    remembers previous turns until `reset` is called.
    """

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are an assistant that helps users solve problems.\n"
        "Always reply in the language of the user's latest message.\n"
        "Use the available tools whenever they can provide information you do "
        "not have, such as the current date or time.\n"
        "Never make up real-time or live information. If no tool can provide "
        "it, say plainly that you cannot know it."
    )

    DEFAULT_MAX_ITERATIONS = 10

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        tools: Sequence[Tool] = (),
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """Initialize the agent with an empty conversation.

        Args:
            client: The Gemini client used to send requests.
            model: The name of the Gemini model, e.g. "gemini-3.5-flash-lite".
            system_instruction: The instruction that frames the model's behavior.
            tools: Optional sequence of tools that the agent can use to answer
                questions.
            max_iterations: Maximum number of model calls in a single turn.

        Raises:
            ValueError: If max_iterations is less than 1.
        """
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")

        self._client = client
        self._model = model
        self._config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=[t.declaration for t in tools])]
            if tools
            else None,
        )
        self._history: list[types.Content] = []
        self._tools = {tool.name: tool for tool in tools}
        self._max_iterations = max_iterations

    @property
    def history(self) -> tuple[types.Content, ...]:
        """Read-only snapshot of the conversation, oldest message first."""
        return tuple(self._history)

    def chat(self, user_prompt: str) -> str:
        """Send a message to the model and return its reply.

        If the model requests tools, the agent runs them, sends their results
        back, and returns the model's final answer.

        The whole turn (the message, any tool calls and results, and the reply)
        is added to the history only if it completes, so a failed or cancelled
        turn leaves the history unchanged.

        Args:
            user_prompt: The user's message.

        Returns:
            The text of the model's final reply (empty if it has no text).

        Raises:
            RuntimeError: If a request fails, Gemini returns no candidates, or
                the model still requests tools after max_iterations calls.
        """
        user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=user_prompt)]
        )

        turn: list[types.Content] = [user_content]

        for _ in range(self._max_iterations):
            model_content = self._generate([*self._history, *turn])
            turn.append(model_content)

            function_calls = self._function_calls(model_content)
            if not function_calls:
                self._history.extend(turn)
                return self._text(model_content)

            tools_responses = [self._run_tool(call) for call in function_calls]

            turn.append(
                types.Content(
                    role="user",
                    parts=tools_responses,
                )
            )

        raise RuntimeError(
            f"No final answer within the {self._max_iterations}-call limit: "
            "the model kept requesting tools"
        )

    def reset(self) -> None:
        """Start a new conversation by clearing the history."""
        self._history.clear()

    def _generate(self, contents: list[types.Content]) -> types.Content:
        """Send contents to the model and return the model's response."""
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._config,
            )
        except genai.errors.APIError as error:
            raise RuntimeError(f"Gemini request failed: {error}") from error
        except Exception as error:  # network, timeout... the SDK does not wrap them all
            raise RuntimeError(f"Gemini request failed: {error!r}") from error

        if not response.candidates or response.candidates[0].content is None:
            raise RuntimeError(
                "Gemini returned no candidates (response may have been blocked)"
            )

        return response.candidates[0].content

    def _run_tool(self, call: types.FunctionCall) -> types.Part:
        """Run a tool requested by the model.

        This method never raises: an unknown tool or an exception raised by the
        tool is reported to the model as an error, so it can react to it.

        Args:
            call: The function call requested by the model.

        Returns:
            A function response part containing either a "result" or an "error".
        """
        name = call.name or ""
        tool = self._tools.get(name)

        if tool is None:
            response = {
                "error": f"Unknown tool '{name}'. Available tools: {list(self._tools)}"
            }
        else:
            try:
                tool_response = tool.run(**(call.args or {}))
                response = {"result": tool_response}
            except Exception as error:
                response = {"error": f"Tool request failed: {error!r}"}

        return types.Part.from_function_response(name=name, response=response)

    @staticmethod
    def _function_calls(content: types.Content) -> list[types.FunctionCall]:
        """Return the function calls requested in a model message."""
        return [
            part.function_call for part in content.parts or [] if part.function_call
        ]

    @staticmethod
    def _text(content: types.Content) -> str:
        """Return the concatenated text parts of a model message."""
        return "".join(part.text for part in content.parts or [] if part.text)
