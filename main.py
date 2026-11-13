"""Command-line chat with the agent."""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent import Agent
from tools import CalculateTool, GetCurrentTimeTool

EXIT_COMMANDS = {"/quit", "/exit"}
RESET_COMMAND = "/reset"
HISTORY_COMMAND = "/history"
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


def main() -> None:
    """Run an interactive chat session with the agent in the terminal."""
    load_dotenv()

    agent = Agent(
        client=genai.Client(),
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        tools=[CalculateTool(), GetCurrentTimeTool()],
    )

    print("Welcome to simple agent!")
    print("Available Commands:")
    print(" - /reset to start a new conversation")
    print(" - /quit or /exit to end the session")
    print(" - /history to view the conversation history")
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
