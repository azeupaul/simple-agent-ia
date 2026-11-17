# simple-agent-ia

A learning project to understand how an AI agent works by building one **without a framework**: no LangChain, no automatic function calling. Only the [Google Gen AI SDK](https://github.com/googleapis/python-genai) to talk to Gemini, and plain Python for everything else.

## Features

- Interactive chat in the terminal
- Conversation memory: the whole history is sent to the model on each turn
- Tool calling, handled manually: the model requests a tool, the agent runs it and sends the result back
- Agent loop: the model can chain several tool calls in one turn, up to a maximum number of iterations
- Parallel tool calls in a single response
- Tool errors are reported to the model, which can correct itself and retry
- Conversation inspection with `/history`
- Live tracing of the agent loop with `/trace`: model calls, tool calls, results and durations
- Human confirmation: sensitive tools run only after the user approves the exact call

### Available tools

| Tool | Description |
|---|---|
| `get_current_time` | Current date and time in a given IANA timezone |
| `write_note` | Saves a note as a Markdown file in `notes/`. **Requires confirmation.** |
| `calculate` | Arithmetic evaluator (`+ - * / // % **`, parentheses). It parses the expression with `ast` and never calls `eval`, and it rejects results that would be too large to compute or send back. |

## How it works

```
user   → "How many hours are left before midnight in Tokyo?"
model  → function_call get_current_time(timezone="Asia/Tokyo")
agent  → runs it, sends back "2026-11-13T09:03:59+09:00"
model  → function_call calculate(expression="24 - (9 + 3 / 60)")
agent  → runs it, sends back 14.95
model  → "About 14 hours and 57 minutes."
```

The model never runs anything itself: it only **requests** a tool, based on the tool's name, description and JSON Schema. The agent executes it, sends the result back, and loops until the model answers with text. `max_iterations` (10 by default) caps the number of model calls in a turn.

| File | Role |
|---|---|
| `agent.py` | `Agent`: keeps the history, calls Gemini, runs the requested tools |
| `tools.py` | `Tool` base class, `@tool` decorator, `ToolRegistry` and the concrete tools |
| `main.py` | Configuration and the terminal chat loop |

A turn is committed to the history only once it completes, so an API error or a `Ctrl+C` leaves the conversation unchanged.

### Tracing

The agent announces each step of its loop to an **observer** (`AgentObserver`), without knowing who listens. In the terminal, `/trace` turns on a `ConsoleTracer`:

```
    --> model call #1
    --> get_current_time(timezone='Asia/Tokyo')
    <-- {'result': '2026-11-17T01:35:06+09:00'} (7.0 ms)
    --> model call #2
    --> calculate(expression='24 * 60 - (1 * 60 + 35)')
    <-- {'result': 1345} (0.2 ms)
    --> model call #3
    Answered after 3 model call(s)
```

Tracing shows the cost and the path of an answer, not only the answer: how many model calls it took, and which tools were called with which arguments.

### Confirmation of sensitive tools

A tool marked `requires_confirmation` runs only if the user approves the exact call, arguments included:

```
The model wants to run write_note(title='Meeting Thursday', content='Meeting on Thursday at 10.')
Allow it? [y/N]
```

Anything other than `y` or `yes` refuses, and the model is told that the user declined. An agent created without a `confirm` function refuses every sensitive tool: forgetting the configuration closes access instead of opening it.

`write_note` also protects itself, since the model chooses the file name: titles are limited to letters, digits, spaces, `_` and `-`, the final path must stay in `notes/`, an existing note is never overwritten, and the content is limited to 10,000 characters.

## Getting started

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/) and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone git@github.com:azeupaul/simple-agent-ia.git
cd simple-agent-ia
uv sync
cp .env.example .env   # then set your API key in .env
uv run main.py
```

| Variable | Required | Default |
|---|---|---|
| `GEMINI_API_KEY` | yes | |
| `GEMINI_MODEL` | no | `gemini-3.5-flash-lite` |

### Commands

| Command | Action |
|---|---|
| `/history` | Show the conversation, including tool calls and results |
| `/trace` | Turn tracing of the agent loop on or off |
| `/reset` | Start a new conversation |
| `/quit`, `/exit`, `Ctrl+D` | Leave |
| `Ctrl+C` | Cancel the current request, or leave at the prompt |

## Adding a tool

### With the `@tool` decorator

For a simple tool, write a function with type annotations and a Google-style docstring:

```python
@tool
def get_current_time(timezone: str) -> str:
    """Get the current date and time in a given timezone.

    Use it whenever the user asks for the time or date somewhere.

    Args:
        timezone: IANA timezone name, e.g. "Europe/Paris".
    """
    return datetime.now(ZoneInfo(timezone)).isoformat()
```

For a tool with side effects, use `@tool(requires_confirmation=True)`: the user will be asked to approve each call.

The decorator builds the tool from the function itself:

| Tool field | Comes from |
|---|---|
| `name` | the function's name |
| `description` | the docstring, before the first section |
| `parameters` | the type annotations, the defaults (a parameter without a default is required) and the `Args:` section |

Supported parameter types are `str`, `int`, `float` and `bool`. The tool is refused when it starts if a parameter has no annotation or no description, or if the docstring documents a parameter the function does not have, so the description sent to the model always matches the code.

### With a `Tool` subclass

For a tool that needs state or helper methods, like `calculate`, subclass `Tool`, define its three class attributes and implement `run`:

```python
class MyTool(Tool):
    """Short description of the tool."""

    name = "my_tool"
    description = "What the tool does and when the model should use it."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"arg": {"type": "string", "description": "..."}},
        "required": ["arg"],
    }

    def run(self, **kwargs: Any) -> str:
        """Run the tool."""
        return kwargs["arg"]
```

### Registering it

Add it to the agent in `main.py`. A decorated function is already a tool instance, so it takes no parentheses:

```python
Agent(..., tools=[CalculateTool(), get_current_time, MyTool()])
```

Tool names must be unique: the agent refuses two tools with the same name.

The `description` is a prompt: it is the only thing the model reads to decide when to use the tool. Say what the tool accepts and what it does not, so the model does not waste iterations on rejected calls.

Treat the arguments as untrusted input: they come from the model, which can be manipulated. Validate them, and never pass them to `eval`, a shell or a file path without checks.

If `run` raises an exception, the agent catches it and sends the error message to the model.

## Development

```bash
uv run pytest                     # run the tests
uv run pre-commit install         # once: run the checks before each commit
uv run pre-commit run --all-files # run them manually
```

The checks are `ruff format`, `ruff check`, `mypy` in strict mode and `pytest`. GitHub Actions runs the same hooks on every push and pull request.

The tests never call Gemini: `tests/test_agent.py` replaces the client with a fake one that replays scripted responses, so they are free, fast and deterministic.

## Roadmap

- [x] Conversation memory and terminal chat
- [x] Tools with manual function calling
- [x] Agent loop: chain several tool calls, with a maximum number of iterations
- [x] Tool registry and `@tool` decorator
- [x] Tracing and human confirmation for sensitive tools
- [ ] Going further: history summarization, planning, structured output, streaming

## Known limitations

- The model may request a call again after the user declined it in the same turn. Each run still needs a new approval, so nothing runs without consent, but the user may have to say no twice.
- Note titles are case-sensitive on Linux: `Meeting` and `meeting` create two notes, while macOS and Windows treat them as the same file and refuse the second one.

## License

[MIT](LICENSE)
