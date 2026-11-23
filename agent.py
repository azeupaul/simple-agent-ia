"""Conversational agent backed by a Gemini model."""

import time
from collections.abc import Callable, Generator, Sequence
from typing import Any

from google import genai
from google.genai import types

from tools import Tool, ToolRegistry


def _deny(call: types.FunctionCall) -> bool:
    """Refuse every sensitive tool: the default when no confirm is given."""
    return False


class AgentObserver:
    """Receives the agent's events. Every method does nothing by default."""

    def on_model_request(self, iteration: int) -> None:
        """Called before each call to the model (iteration starts at 1)."""

    def on_tool_call(self, call: types.FunctionCall) -> None:
        """Called before a tool runs."""

    def on_tool_result(
        self, name: str, response: dict[str, Any], duration: float
    ) -> None:
        """Called after a tool runs, with its result or error and its duration."""

    def on_final_answer(self, iterations: int) -> None:
        """Called when the model answers with text, after this many model calls."""

    def on_usage(self, prompt_tokens: int, output_tokens: int) -> None:
        """Called after each model call, with the tokens it consumed."""

    def on_history_summarized(self, summarized: int, kept: int) -> None:
        """Called when old messages were replaced by a summary."""

    def on_summary_failed(self, error: Exception) -> None:
        """Called when the summary failed: the history is kept unchanged."""


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

    SUMMARY_INSTRUCTION = (
        "You summarize a conversation between a user and an assistant, so that "
        "the assistant can go on with it without the original messages.\n"
        "Keep every fact, decision, name, number and open question that may "
        "matter later, including what the tools returned. Leave out greetings "
        "and small talk.\n"
        "Write in the language of the conversation, as short paragraphs or "
        "bullet points. Reply with the summary only."
    )

    SUMMARY_PREFIX = "Summary of the earlier conversation:\n"

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        tools: Sequence[Tool] = (),
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        observer: AgentObserver | None = None,
        confirm: Callable[[types.FunctionCall], bool] | None = None,
        max_context_tokens: int | None = None,
        keep_recent_turns: int = 2,
    ) -> None:
        """Initialize the agent with an empty conversation.

        Args:
            client: The Gemini client used to send requests.
            model: The name of the Gemini model, e.g. "gemini-3.5-flash-lite".
            system_instruction: The instruction that frames the model's behavior.
            tools: Optional sequence of tools that the agent can use to answer
                questions.
            max_iterations: Maximum number of model calls in a single turn.
            observer: Receives the agent's events, e.g. to trace them.
                Defaults to an observer that does nothing.
            confirm: Asks the user whether a sensitive tool may run, and returns True
                to allow it. By default, every sensitive tool is refused.
            max_context_tokens: When the conversation grows beyond this many
                tokens, its older turns are summarized after the current turn.
                None (the default) never summarizes.
            keep_recent_turns: Number of most recent turns kept as they are when
                the history is summarized.

        Raises:
            ValueError: If max_iterations, max_context_tokens (when given) or
                keep_recent_turns is less than 1.
        """
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")

        if max_context_tokens is not None and max_context_tokens < 1:
            raise ValueError(
                "max_context_tokens must be at least 1 when defined, "
                f"got {max_context_tokens}"
            )

        if keep_recent_turns < 1:
            raise ValueError(
                f"keep_recent_turns must be at least 1, got {keep_recent_turns}"
            )

        self._client = client
        self._model = model
        self._tools = ToolRegistry(tools)
        self._config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=self._tools.declarations)]
            if self._tools
            else None,
        )
        self._history: list[types.Content] = []
        self._max_iterations = max_iterations
        self._observer = observer or AgentObserver()
        self._confirm = confirm or _deny
        self._context_tokens: int = 0
        self._max_context_tokens = max_context_tokens
        self._keep_recent_turns = keep_recent_turns

    @property
    def history(self) -> tuple[types.Content, ...]:
        """Read-only snapshot of the conversation, oldest message first."""
        return tuple(self._history)

    @property
    def context_tokens(self) -> int:
        """Approximate size of the conversation, in tokens, after the last model call.

        It is the prompt sent on the last call (history, instructions and tool
        declarations) plus the reply. 0 before the first call.
        """
        return self._context_tokens

    def chat_stream(self, user_prompt: str) -> Generator[str, None, None]:
        """Send a message to the model and stream its reply as it is written.

        If the model requests tools, the agent runs them, sends their results
        back and keeps streaming, until the model answers without requesting
        any tool. The stream contains all the text the model writes during the
        turn, including any text written before a tool call.

        The whole turn (the message, any tool calls and results, and the reply)
        is added to the history only once the stream is fully consumed: a
        failed turn, or a stream the caller stops early (e.g. with Ctrl+C),
        leaves the history unchanged.

        Args:
            user_prompt: The user's message.

        Yields:
            Pieces of the model's text, in order, as they arrive.

        Raises:
            RuntimeError: If a request fails, Gemini returns no candidates, or
                the model still requests tools after max_iterations calls.
        """
        user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=user_prompt)]
        )

        turn: list[types.Content] = [user_content]

        for iteration in range(1, self._max_iterations + 1):
            self._observer.on_model_request(iteration)

            model_content = yield from self._stream([*self._history, *turn])
            turn.append(model_content)

            function_calls = self._function_calls(model_content)
            if not function_calls:
                self._history.extend(turn)

                self._observer.on_final_answer(iteration)

                if self._should_summarize():
                    self._summarize_history()
                return

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

    def chat(self, user_prompt: str) -> str:
        """Send a message to the model and return its whole reply at once.

        Same as ``chat_stream``, but waits for the end of the turn and returns
        all the text the model wrote, joined.

        Args:
            user_prompt: The user's message.

        Returns:
            All the text the model wrote during the turn (empty if none).

        Raises:
            RuntimeError: If a request fails, Gemini returns no candidates, or
                the model still requests tools after max_iterations calls.
        """
        return "".join(self.chat_stream(user_prompt))

    def reset(self) -> None:
        """Start a new conversation by clearing the history."""
        self._history.clear()
        self._context_tokens = 0

    def _stream(
        self, contents: list[types.Content]
    ) -> Generator[str, None, types.Content]:
        """Stream the model's response to the contents.

        Yields:
            The text of the response, piece by piece, as it arrives.

        Returns:
            The complete model message. Its parts are kept exactly as received,
            so that Gemini's thought signatures stay where the API expects them.

        Raises:
            RuntimeError: If the request fails, or if Gemini returns nothing
                (the response may have been blocked).
        """
        parts: list[types.Part] = []
        usage = None
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=self._config,
            ):
                if chunk.usage_metadata:
                    usage = chunk.usage_metadata

                if (
                    not chunk.candidates
                    or chunk.candidates[0].content is None
                    or not chunk.candidates[0].content.parts
                ):
                    continue

                for part in chunk.candidates[0].content.parts:
                    parts.append(part)
                    if part.text:
                        yield part.text
        except genai.errors.APIError as error:
            raise RuntimeError(f"Gemini request failed: {error}") from error
        except Exception as error:  # network, timeout... the SDK does not wrap them all
            raise RuntimeError(f"Gemini request failed: {error!r}") from error

        if usage:
            prompt_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            self._observer.on_usage(
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            # Not total_token_count: it also counts thinking tokens, which are
            # billed but not kept in the history.
            self._context_tokens = prompt_tokens + output_tokens

        if not parts:
            raise RuntimeError(
                "Gemini returned no candidates (response may have been blocked)"
            )

        return types.Content(role="model", parts=parts)

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

        self._observer.on_tool_call(call)
        duration = 0.0

        if tool is None:
            response = {
                "error": f"Unknown tool '{name}'. Available tools: {self._tools.names}"
            }
        elif tool.requires_confirmation and not self._confirm(call):
            response = {
                "error": "The user declined to run this tool. "
                "Do not retry it unless the user asks."
            }
        else:
            start = time.perf_counter()
            try:
                tool_response = tool.run(**(call.args or {}))
                response = {"result": tool_response}
            except Exception as error:
                response = {"error": f"Tool request failed: {error!r}"}
            duration = time.perf_counter() - start

        self._observer.on_tool_result(name, response, duration)

        return types.Part(
            function_response=types.FunctionResponse(
                id=call.id, name=name, response=response
            )
        )

    @staticmethod
    def _function_calls(content: types.Content) -> list[types.FunctionCall]:
        """Return the function calls requested in a model message."""
        return [
            part.function_call for part in content.parts or [] if part.function_call
        ]

    @staticmethod
    def _turn_starts(history: Sequence[types.Content]) -> list[int]:
        """Return the index of the first message of each turn.

        A turn starts with a user message that contains text. Tool results are
        also sent with the "user" role, but they only contain function responses.
        """
        return [
            index
            for index, content in enumerate(history)
            if content.role == "user" and any(part.text for part in content.parts or [])
        ]

    def _split_history(self) -> tuple[list[types.Content], list[types.Content]]:
        """Split the history into old turns to summarize and recent turns to keep.

        The cut is always at the start of a turn, so a tool call is never
        separated from its result.

        Returns:
            The old messages and the recent ones. The old part is empty when
            the history has no more than ``keep_recent_turns`` turns.
        """
        starts = self._turn_starts(self._history)
        if len(starts) <= self._keep_recent_turns:
            return [], list(self._history)

        cut = starts[-self._keep_recent_turns]
        return self._history[:cut], self._history[cut:]

    def _should_summarize(self) -> bool:
        """Return True if the conversation has grown beyond max_context_tokens."""
        return (
            self._max_context_tokens is not None
            and self._context_tokens > self._max_context_tokens
        )

    @staticmethod
    def _transcript(contents: Sequence[types.Content]) -> str:
        """Render messages as plain text, one line per text, tool call or result.

        Empty texts (the parts that only carry a thought signature) are skipped.
        """
        lines = []
        for content in contents:
            speaker = "Assistant" if content.role == "model" else "User"
            for part in content.parts or []:
                if part.text:
                    lines.append(f"{speaker}: {part.text}")
                elif part.function_call:
                    call = part.function_call
                    lines.append(f"Tool call: {call.name}({call.args or {}})")
                elif part.function_response:
                    result = part.function_response
                    lines.append(f"Tool result: {result.name} -> {result.response}")
        return "\n".join(lines)

    def _summarize_history(self) -> None:
        """Replace the old turns of the history with a summary.

        The old turns are sent to the model as a plain-text transcript, in a
        separate request without tools, and replaced by a pair of messages: the
        summary, then a short acknowledgement from the model. The most recent
        turns are kept as they are.

        This method never raises: the user's turn is already complete, so if
        the summary fails, the history is kept unchanged and the summary will
        be tried again after the next turn.
        """
        old, recent = self._split_history()
        if not old:
            return

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=self._transcript(old),
                config=types.GenerateContentConfig(
                    system_instruction=self.SUMMARY_INSTRUCTION
                ),
            )
            summary = response.text
        except Exception as error:  # APIError, network... the turn must not fail
            self._observer.on_summary_failed(error)
            return

        if not summary:
            self._observer.on_summary_failed(
                RuntimeError("Gemini returned an empty summary")
            )
            return

        self._history = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self.SUMMARY_PREFIX + summary)],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(text="Understood, I will keep it in mind.")
                ],
            ),
            *recent,
        ]
        self._observer.on_history_summarized(summarized=len(old), kept=len(recent))
