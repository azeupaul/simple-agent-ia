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

### Available tools

| Tool | Description |
|---|---|
| `get_current_time` | Current date and time in a given IANA timezone |
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
| `tools.py` | `Tool` abstract base class and the concrete tools |
| `main.py` | Configuration and the terminal chat loop |

A turn is committed to the history only once it completes, so an API error or a `Ctrl+C` leaves the conversation unchanged.

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
| `/reset` | Start a new conversation |
| `/quit`, `/exit`, `Ctrl+D` | Leave |
| `Ctrl+C` | Cancel the current request, or leave at the prompt |

## Adding a tool

Subclass `Tool`, define its three class attributes and implement `run`:

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

Then register it in `main.py`: `Agent(..., tools=[CalculateTool(), GetCurrentTimeTool(), MyTool()])`.

The `description` is a prompt: it is the only thing the model reads to decide when to use the tool. Say what the tool accepts and what it does not, so the model does not waste iterations on rejected calls.

Treat the arguments as untrusted input: they come from the model, which can be manipulated. Validate them, and never pass them to `eval`, a shell or a file path without checks.

If `run` raises an exception, the agent catches it and sends the error message to the model.

## Development

```bash
uv run pre-commit install         # once: run the checks before each commit
uv run pre-commit run --all-files # run them manually
```

The checks are `ruff format`, `ruff check` and `mypy` in strict mode. GitHub Actions runs the same hooks on every push and pull request.

## Roadmap

- [x] Conversation memory and terminal chat
- [x] Tools with manual function calling
- [x] Agent loop: chain several tool calls, with a maximum number of iterations
- [ ] Tool registry
- [ ] Tracing and human confirmation for sensitive tools
- [ ] Going further: history summarization, planning, structured output, streaming

## License

[MIT](LICENSE)
