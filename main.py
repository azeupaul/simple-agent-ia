from agent import call_agent

EXIT_COMMANDS = {"quit", "exit", "q"}

def main():
    print("Welcome to simple agent... [type quit, exit or q to exit the program]")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()  # newline so the goodbye isn't glued to the prompt
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            break

        try:
            response = call_agent(user_prompt=user_input)
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
