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

    DEFAULT_SYSTEM_INSTRUCTION = "Tu es un assistant qui aide à résoudre des problèmes."

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        tools: Sequence[Tool] = (),
    ) -> None:
        """Initialize the agent with an empty conversation.

        Args:
            client: The Gemini client used to send requests.
            model: The name of the Gemini model, e.g. "gemini-3.5-flash-lite".
            system_instruction: The instruction that frames the model's behavior.
            tools: Optional sequence of tools that the agent can use to answer
                questions.
        """
        self._client = client
        self._model = model
        declared_tools = (
            [types.Tool(function_declarations=[tool.declaration for tool in tools])]
            if tools
            else None
        )
        self._config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=declared_tools,
        )
        self._history: list[types.Content] = []
        self._tools = {tool.name: tool for tool in tools}

    @property
    def history(self) -> tuple[types.Content, ...]:
        """Read-only snapshot of the conversation, oldest message first."""
        return tuple(self._history)

    def chat(self, user_prompt: str) -> str:
        """Send a message to the model and return its reply.

        The message and the reply are added to the history only if the request
        succeeds, so a failed or cancelled request leaves the history unchanged.

        Args:
            user_prompt: The user's message.

        Returns:
            The text of the model's reply (empty if the reply has no text).

        Raises:
            RuntimeError: If the request fails or Gemini returns no candidates.
        """
        user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=user_prompt)]
        )
        contents = [*self._history, user_content]

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

        if response.function_calls:
            for call in response.function_calls:
                print(f"[model requests] {call.name}({call.args})")
            return ""

        self._history.extend([user_content, response.candidates[0].content])

        return response.text or ""

    def reset(self) -> None:
        """Start a new conversation by clearing the history."""
        self._history.clear()
