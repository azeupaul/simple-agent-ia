# simple-agent-ia

A learning project to understand how an AI agent works by building one **without a framework**: no LangChain, no automatic function calling. Only the [Google Gen AI SDK](https://github.com/googleapis/python-genai) to talk to Gemini, and plain Python for everything else.

## Features

- Interactive chat in the terminal, with the reply streamed as it is written
- Conversation memory: the whole history is sent to the model on each turn
- Tool calling, handled manually: the model requests a tool, the agent runs it and sends the result back
- Agent loop: the model can chain several tool calls in one turn, up to a maximum number of iterations
- Parallel tool calls in a single response
- Tool errors are reported to the model, which can correct itself and retry
- Conversation inspection with `/history`
- Live tracing of the agent loop with `/trace`: model calls, tool calls, results, durations and tokens
- Conversation size in tokens with `/tokens`
- History summarization: beyond a token threshold, the older turns are replaced by a summary
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
| `main.py` | Configuration, the terminal chat loop and `Console`, which prints the streamed reply, traces and confirmations without mixing them |

A turn is committed to the history only once it completes, so an API error or a `Ctrl+C` leaves the conversation unchanged.

### Streaming

`Agent.chat_stream()` is a generator: it yields the model's text piece by piece, as Gemini sends it, while the agent loop runs tool calls in between. `Agent.chat()` simply joins the stream.

```python
for piece in agent.chat_stream("What time is it in Tokyo?"):
    print(piece, end="", flush=True)
```

- The stream contains all the text the model writes during the turn, including any text written before a tool call: the agent cannot know in advance whether a call will follow.
- A turn is committed only once the stream is fully consumed. If the stream fails midway, or the caller stops reading it (`Ctrl+C`), the history is left unchanged.
- The parts of each model message are stored **exactly as received**. Gemini 3 attaches *thought signatures* to some parts, sometimes to a last part with empty text, and the API requires them back in the same place. Merging the streamed text into one part would break that, so the pieces are joined only for display, in `/history`.
- Tool results are sent back with the `id` of the call they answer.

### Tracing

The agent announces each step of its loop to an **observer** (`AgentObserver`), without knowing who listens. In the terminal, `/trace` turns on a `ConsoleTracer`:

```
    --> model call #1
    <-- 546 tokens in, 22 out
    --> get_current_time(timezone='Africa/Douala')
    <-- {'result': '2026-11-23T16:09:24.951183+01:00'} (0.1 ms)
    --> model call #2
    <-- 614 tokens in, 19 out
    Answered after 2 model call(s)
```

Tracing shows the cost and the path of an answer, not only the answer: how many model calls it took, which tools were called with which arguments, and how many tokens each call consumed.

The "in" count of a call is the whole prompt: the system instruction, the tool declarations and the history. It grows on every turn, because the whole conversation is sent again each time. The "out" count leaves out thinking tokens, which are billed but not kept in the history. `/tokens` shows the size of the conversation after the last call: its last prompt plus its reply.

### History summarization

Since the whole history is sent on every turn, a long conversation gets slower, costs more, and would end up beyond the model's context window. When `MAX_CONTEXT_TOKENS` is set and the conversation grows beyond it, the agent replaces its older turns with a summary, at the end of the turn:

```
before:  [turn 1] [turn 2] [turn 3] [turn 4] [turn 5]
after:   [summary of turns 1-3] [turn 4] [turn 5]
```

- The cut is always at the start of a turn, so a tool call is never separated from its result. The most recent turns, 2 by default (`keep_recent_turns`), are kept as they are.
- The old turns are sent to the model as a plain-text transcript, in a separate request without tools, and replaced by two messages: the summary, as a user message, and a short acknowledgement from the model.
- The summary counts as a turn. At the next summary, it becomes one of the old turns and is summarized again with them: the history always holds a single summary, never a pile of them.
- The user's turn is already complete when the summary runs, so a failed summary never fails it. The history is kept unchanged, and the summary is tried again after the next turn.

With `/trace`, a summary shows as:

```
    ~~> history summarized: 10 messages replaced, 6 kept
```

The threshold must be well above what a summary cannot remove: the fixed cost of the system instruction and tool declarations (about 500 tokens here) and the turns that are kept. Otherwise the conversation stays beyond it, and the agent summarizes again on every turn.

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
| `MAX_CONTEXT_TOKENS` | no | unset: the history is never summarized |

The chat refuses to start if `MAX_CONTEXT_TOKENS` is not a positive integer.

### Commands

| Command | Action |
|---|---|
| `/history` | Show the conversation, including tool calls and results |
| `/trace` | Turn tracing of the agent loop on or off |
| `/tokens` | Show the size of the conversation in tokens |
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

The tests never call Gemini: `tests/test_agent.py` replaces the client with a fake one that replays scripted streams, including failures in the middle of a stream, so they are free, fast and deterministic.

## Roadmap

- [x] Conversation memory and terminal chat
- [x] Tools with manual function calling
- [x] Agent loop: chain several tool calls, with a maximum number of iterations
- [x] Tool registry and `@tool` decorator
- [x] Tracing and human confirmation for sensitive tools
- [ ] Going further
  - [x] Streaming
  - [x] History summarization
  - [ ] Structured output
  - [ ] Planning

## Known limitations

- The model may request a call again after the user declined it in the same turn. Each run still needs a new approval, so nothing runs without consent, but the user may have to say no twice.
- Note titles are case-sensitive on Linux: `Meeting` and `meeting` create two notes, while macOS and Windows treat them as the same file and refuse the second one.
- The recent turns are kept by number, not by size. If they alone are beyond `MAX_CONTEXT_TOKENS`, for instance after a very long answer, a summary cannot bring the conversation back under it, and the agent summarizes again on every turn until that answer becomes an old turn.
- The summary request is not counted in the trace's token counts, although it is billed.
- Each summary of a summary can lose a little detail: what the model left out of the first summary is gone for good.

## License

[MIT](LICENSE)
