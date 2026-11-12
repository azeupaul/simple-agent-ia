"""Command-line chat with the agent."""

import os

from dotenv import load_dotenv
from google import genai

from agent import Agent
from tools import CalculateTool, GetCurrentTimeTool

EXIT_COMMANDS = {"/quit", "/exit"}
RESET_COMMAND = "/reset"


def main() -> None:
    """Run an interactive chat session with the agent in the terminal."""
    load_dotenv()

    agent = Agent(
        client=genai.Client(),
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        tools=[CalculateTool(), GetCurrentTimeTool()],
    )

    print("Welcome to simple agent!")
    print("Commands: /reset to start a new conversation, /quit or /exit to leave.")

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
