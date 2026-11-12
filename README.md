# simple-agent-ia

A learning project to understand how an AI agent works by building one **without a framework**: no LangChain, no automatic function calling. Only the [Google Gen AI SDK](https://github.com/googleapis/python-genai) to talk to Gemini, and plain Python for everything else.

## Features

- Interactive chat in the terminal
- Conversation memory: the whole history is sent to the model on each turn
- Tool calling, handled manually: the model requests a tool, the agent runs it and sends the result back
- Parallel tool calls in a single response
- Tool errors are reported to the model instead of crashing the agent

## How it works

```
user   → "What time is it in Paris and Tokyo?"
model  → function_call get_current_time(timezone="Europe/Paris")
         function_call get_current_time(timezone="Asia/Tokyo")
agent  → runs both tools, sends back the two function responses
model  → "It is 01:08 in Paris and 09:08 in Tokyo."
```

The model never runs anything itself: it only **requests** a tool, based on the tool's name, description and JSON Schema. The agent executes it.

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

Then register it in `main.py`: `Agent(..., tools=[GetCurrentTimeTool(), MyTool()])`.

The `description` is a prompt: it is the only thing the model reads to decide when to use the tool.

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
