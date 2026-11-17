"""Command-line chat with the agent."""

import os
from typing import Any, override

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent import Agent, AgentObserver
from tools import CalculateTool, get_current_time, write_note

EXIT_COMMANDS = {"/quit", "/exit"}
RESET_COMMAND = "/reset"
HISTORY_COMMAND = "/history"
TRACER_COMMAND = "/trace"
MAX_DISPLAY_LENGTH = 80


def format_part(role: str | None, part: types.Part) -> str:
    """Return a one-line, human-readable summary of a message part."""
    if part.text:
        text = part.text.replace("\n", " ")
        if len(text) > MAX_DISPLAY_LENGTH:
            text = text[: MAX_DISPLAY_LENGTH - 1] + "…"
        return f"[{role} text]: {text}"
    if part.function_call:
        call = part.function_call
        return f"[{role} call]: {call.name}({call.args})"
    if part.function_response:
        result = part.function_response
        return f"[{role} result]: {result.name} -> {result.response}"
    return f"[{role} unknown part]"


def format_call(call: types.FunctionCall) -> str:
    """Return a one-line, human-readable call to a tool."""
    args = ", ".join(f"{key}={value!r}" for key, value in (call.args or {}).items())
    return f"{call.name}({args})"


class ConsoleTracer(AgentObserver):
    """Print the agent's events in the terminal while tracing is enabled."""

    def __init__(self, *, enabled: bool = False) -> None:
        """Initialize the tracer.

        Args:
            enabled: Whether to print events from the start.
        """
        self.enabled = enabled

    @override
    def on_model_request(self, iteration: int) -> None:
        """Print the number of the model call."""
        if self.enabled:
            print(f"    --> model call #{iteration}")

    @override
    def on_tool_call(self, call: types.FunctionCall) -> None:
        """Print the tool name and its arguments."""
        if self.enabled:
            print(f"    --> {format_call(call)}")

    @override
    def on_tool_result(
        self, name: str, response: dict[str, Any], duration: float
    ) -> None:
        """Print the tool's result or error, and how long it took."""
        if self.enabled:
            result = str(response)
            if len(result) > MAX_DISPLAY_LENGTH:
                result = result[: MAX_DISPLAY_LENGTH - 1] + "…"
            print(f"    <-- {result} ({duration * 1000:.1f} ms)")

    @override
    def on_final_answer(self, iterations: int) -> None:
        """Print how many model calls the answer took."""
        if self.enabled:
            print(f"    Answered after {iterations} model call(s)")


def ask_confirmation(call: types.FunctionCall) -> bool:
    """Ask the user whether the model may run this tool call.

    Ctrl+C is not caught here: it cancels the whole request in main().

    Returns:
        True only if the user answers "y" or "yes"; False otherwise, or on Ctrl+D.
    """
    print(f"The model wants to run {format_call(call)}")

    try:
        answer = input("Allow it? [y/N] ")
    except EOFError:
        print()
        return False

    return answer.strip().lower() in {"y", "yes"}


def main() -> None:
    """Run an interactive chat session with the agent in the terminal."""
    load_dotenv()

    tracer = ConsoleTracer()
    agent = Agent(
        client=genai.Client(),
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        tools=[CalculateTool(), get_current_time, write_note],
        observer=tracer,
        confirm=ask_confirmation,
    )

    print("Welcome to simple agent!")
    print("Available Commands:")
    print(" - /reset to start a new conversation")
    print(" - /quit or /exit to end the session")
    print(" - /history to view the conversation history")
    print(" - /trace to enable or disable traces")
    print("Type your message and press Enter to send it.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()  # newline so the goodbye isn't glued to the prompt
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in EXIT_COMMANDS:
            break

        if command == RESET_COMMAND:
            agent.reset()
            print("[conversation reset]")
            continue

        if command == HISTORY_COMMAND:
            if not agent.history:
                print("[no conversation history]")
            else:
                for content in agent.history:
                    for part in content.parts or []:
                        print(format_part(content.role, part))
            continue
        if command == TRACER_COMMAND:
            tracer.enabled = not tracer.enabled
            print(f"Tracing {'enabled' if tracer.enabled else 'disabled'}.\n")
            continue

        try:
            response = agent.chat(user_prompt=user_input)
        except KeyboardInterrupt:
            print("\n[request cancelled]")
            continue
        except RuntimeError as error:
            print(f"[error] {error}")
            continue

        print(f"Agent: {response}")

    print("Goodbye!")


if __name__ == "__main__":
    main()
