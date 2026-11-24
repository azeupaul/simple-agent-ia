"""Command-line chat with the agent."""

import os
from typing import Any, override

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent import Agent, AgentObserver
from planning import Plan
from tasks import TaskList
from tools import CalculateTool, get_current_time, write_note

EXIT_COMMANDS = {"/quit", "/exit"}
RESET_COMMAND = "/reset"
HISTORY_COMMAND = "/history"
TRACER_COMMAND = "/trace"
TOKEN_COMMAND = "/tokens"
TASKS_COMMAND = "/tasks"
PLAN_COMMAND = "/plan"
MAX_DISPLAY_LENGTH = 80


def shorten(text: str, max_length: int = MAX_DISPLAY_LENGTH) -> str:
    """Return the text, cut with "…" if it is longer than max_length."""
    if len(text) > max_length:
        text = text[: max_length - 1] + "…"

    return text.replace("\n", " ")


def format_call(call: types.FunctionCall) -> str:
    """Return a one-line, human-readable call to a tool."""
    args = ", ".join(f"{key}={value!r}" for key, value in (call.args or {}).items())
    return f"{call.name}({args})"


def format_part(role: str | None, part: types.Part) -> str:
    """Return a one-line, human-readable summary of a message part."""
    if part.text:
        return f"[{role} text]: {shorten(part.text)}"
    if part.function_call:
        call = part.function_call
        return f"[{role} call]: {call.name}({call.args})"
    if part.function_response:
        result = part.function_response
        return f"[{role} result]: {result.name} -> {result.response}"
    return f"[{role} unknown part]"


def format_content(content: types.Content) -> list[str]:
    """Return one line per part of a message, with consecutive texts joined.

    Streaming splits the model's text into many parts; they are joined here
    for display only. The history itself keeps the parts as received.
    """
    lines: list[str] = []
    texts: list[str] = []  # the "basket": pieces of text waiting to be joined

    def flush_texts() -> None:
        """Empty the basket into a single text line."""
        if texts:
            joined = types.Part.from_text(text="".join(texts))
            lines.append(format_part(content.role, joined))
            texts.clear()

    for part in content.parts or []:
        if part.text:
            texts.append(part.text)
        elif part.function_call or part.function_response:
            flush_texts()
            lines.append(format_part(content.role, part))
        # Any other part (e.g. empty text carrying a thought signature) is skipped.

    flush_texts()

    return lines


class Console:
    """Terminal output that keeps the streamed reply apart from other lines."""

    def __init__(self) -> None:
        """Start with no reply line open."""
        self._replying = False  # True while a reply line is open

    def reply(self, piece: str) -> None:
        """Print a piece of the agent's reply, starting the line if needed."""
        if not self._replying:
            print("Agent: ", end="")
            self._replying = True
        print(piece, end="", flush=True)

    def end_reply(self) -> None:
        """Close the reply line, if one is open."""
        if self._replying:
            print()
            self._replying = False

    def line(self, text: str) -> None:
        """Print a full line, on its own, below any open reply."""
        self.end_reply()
        print(text)

    def ask(self, question: str) -> bool:
        """Ask a yes/no question. Anything but "y" or "yes" means no."""
        try:
            answer = input(f"{question} [y/N] ")
        except EOFError:
            self.line("")
            return False

        return answer.strip().lower() in {"y", "yes"}

    def confirm(self, call: types.FunctionCall) -> bool:
        """Ask the user whether the model may run this tool call."""
        self.line(f"The model wants to run {format_call(call)}")
        return self.ask("Allow it?")


class ConsoleTracer(AgentObserver):
    """Print the agent's events in the terminal while tracing is enabled."""

    def __init__(self, console: Console, *, enabled: bool = False) -> None:
        """Initialize the tracer.

        Args:
            console: Where to print the events.
            enabled: Whether to print events from the start.
        """
        self._console = console
        self.enabled = enabled

    @override
    def on_model_request(self, iteration: int) -> None:
        """Print the number of the model call."""
        if self.enabled:
            self._console.line(f"    --> model call #{iteration}")

    @override
    def on_tool_call(self, call: types.FunctionCall) -> None:
        """Print the tool name and its arguments."""
        if self.enabled:
            self._console.line(f"    --> {shorten(format_call(call))}")

    @override
    def on_tool_result(
        self, name: str, response: dict[str, Any], duration: float
    ) -> None:
        """Print the tool's result or error, and how long it took."""
        if self.enabled:
            result = str(response)
            self._console.line(f"    <-- {shorten(result)} ({duration * 1000:.1f} ms)")

    @override
    def on_final_answer(self, iterations: int) -> None:
        """Print how many model calls the answer took."""
        if self.enabled:
            self._console.line(f"    Answered after {iterations} model call(s)")

    @override
    def on_usage(self, prompt_tokens: int, output_tokens: int) -> None:
        """Print the tokens sent and received by the model call."""
        if self.enabled:
            self._console.line(
                f"    <-- {prompt_tokens} tokens in, {output_tokens} out"
            )

    @override
    def on_history_summarized(self, summarized: int, kept: int) -> None:
        """Print the number of messages replaced and kept after a summary."""
        if self.enabled:
            self._console.line(
                f"    ~~> history summarized: {summarized} messages replaced, "
                f"{kept} kept"
            )

    @override
    def on_summary_failed(self, error: Exception) -> None:
        """Print why the summary failed: the history is kept unchanged."""
        if self.enabled:
            self._console.line(
                f"    ~~> summary failed: {error}, history kept unchanged"
            )

    @override
    def on_invalid_reply(self, attempt: int, problems: str) -> None:
        """Print why a structured reply was refused, before the next attempt."""
        if self.enabled:
            self._console.line(
                f"    ~~> invalid reply (attempt {attempt}): {shorten(problems)}"
            )


def read_max_context_tokens() -> int | None:
    """Return MAX_CONTEXT_TOKENS from the environment, or None if it is unset.

    Raises:
        ValueError: If the variable is set but is not an integer.
    """
    raw = os.getenv("MAX_CONTEXT_TOKENS", "").strip()
    if not raw:
        return None

    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(
            f"MAX_CONTEXT_TOKENS must be an integer, got {raw!r}"
        ) from error


def show_tasks(agent: Agent, console: Console) -> None:
    """Extract the tasks from the conversation and print them."""
    if not agent.history:
        console.line("[no conversation yet]")
        return

    try:
        task_list = agent.extract(TaskList.INSTRUCTION, TaskList)
    except KeyboardInterrupt:
        console.line("[request cancelled]")
        return
    except RuntimeError as error:
        console.line(f"[error] {error}")
        return

    if not task_list.tasks:
        console.line("[no tasks found]")
        return

    console.line(f"Tasks ({len(task_list.tasks)}):")
    for line in task_list.lines():
        console.line(f"  {line}")


def run_plan(agent: Agent, console: Console, request: str) -> None:
    """Plan the request, ask the user to approve the plan, then run it.

    Each step is a normal turn of the conversation, so the model can call
    tools, and sensitive tools still need confirmation.

    Raises:
        RuntimeError: If planning or a step fails. The steps already done
            stay in the history.
    """
    plan: Plan = agent.plan(request)

    console.line(f"Plan: {plan.goal}")
    for line in plan.lines():
        console.line(f"  {line}")
    if not console.ask("Run this plan?"):
        console.line("[plan dropped]")
        return

    total = len(plan.steps)
    for index, step in enumerate(plan.steps, start=1):
        console.line(f"[step {index}/{total}] {step.description}")
        for piece in agent.chat_stream(plan.step_prompt(index)):
            console.reply(piece)
        console.end_reply()

    console.line("[final answer]")
    for piece in agent.chat_stream(plan.final_prompt()):
        console.reply(piece)
    console.end_reply()


def main() -> None:
    """Run an interactive chat session with the agent in the terminal."""
    load_dotenv()

    console = Console()
    tracer = ConsoleTracer(console=console)
    try:
        agent = Agent(
            client=genai.Client(),
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            tools=[CalculateTool(), get_current_time, write_note],
            observer=tracer,
            confirm=console.confirm,
            max_context_tokens=read_max_context_tokens(),
        )
    except ValueError as error:
        raise SystemExit(f"[config error] {error}") from error

    console.line("Welcome to simple agent!")
    console.line("Available Commands:")
    console.line(" - /history to view the conversation history")
    console.line(" - /trace to enable or disable traces")
    console.line(" - /tokens to show the size of the conversation in tokens")
    console.line(" - /tasks to list the tasks mentioned in the conversation")
    console.line(" - /plan <request> to plan a request, then run it step by step")
    console.line(" - /reset to start a new conversation")
    console.line(" - /quit or /exit to end the session")
    console.line("Type your message and press Enter to send it.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.line("")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in EXIT_COMMANDS:
            break

        if command == RESET_COMMAND:
            agent.reset()
            console.line("[conversation reset]")
            continue

        if command == HISTORY_COMMAND:
            if not agent.history:
                console.line("[no conversation history]")
            else:
                for content in agent.history:
                    for line in format_content(content):
                        console.line(line)
            continue

        if command == TRACER_COMMAND:
            tracer.enabled = not tracer.enabled
            console.line(f"Tracing {'enabled' if tracer.enabled else 'disabled'}.")
            continue

        if command == TOKEN_COMMAND:
            console.line(f"Conversation size: {agent.context_tokens} tokens.")
            continue

        if command == TASKS_COMMAND:
            show_tasks(agent, console)
            continue

        if command == PLAN_COMMAND:
            console.line("Usage: /plan <request>")
            continue

        if command.startswith(f"{PLAN_COMMAND} "):
            request = user_input[len(PLAN_COMMAND) :].strip()
            try:
                run_plan(agent, console, request)
            except KeyboardInterrupt:
                console.line("[plan cancelled]")
            except RuntimeError as error:
                console.line(f"[error] {error}")
            continue

        try:
            for piece in agent.chat_stream(user_input):
                console.reply(piece)
            console.end_reply()
        except KeyboardInterrupt:
            console.line("[request cancelled]")
            continue
        except RuntimeError as error:
            console.line(f"[error] {error}")
            continue

    console.line("Goodbye!")


if __name__ == "__main__":
    main()
