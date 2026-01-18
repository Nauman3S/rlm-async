# RLM - Recursive Language Model

A Python implementation of [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab - MIT, 2025) for processing arbitrarily large contexts.

## How It Works

Instead of cramming huge contexts into one LLM call, RLM:

1. **Stores context externally** in a Python sandbox (never sent directly to LLM)
2. **LLM uses tools** to inspect, search, and process the context
3. **Recursively decomposes** large contexts via parallel `sub_llm` calls
4. **Combines results** and returns a final answer

```
┌─────────────────────────────────────────────┐
│  Large Context (100K+ chars)                │
│         ↓                                   │
│  Sandbox (context stored as variable)       │
│         ↓                                   │
│  Root LLM (smart model) uses tools:         │
│    - peek, grep, chunk                      │
│    - sub_llm (parallel recursive calls)     │
│         ↓                                   │
│  Worker LLMs process chunks in parallel     │
│         ↓                                   │
│  Final Answer                               │
└─────────────────────────────────────────────┘
```

## Installation

```bash
git clone <repo>
cd rlm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your OpenRouter API key in `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

## Usage

### CLI

```bash
# Basic usage
python main.py "Find all error messages" --file server.log

# With verbose output (shows trace, stats, tool calls)
python main.py "Summarize key points" --file document.txt -v

# Custom models (smarter root, cheaper workers)
python main.py "Find X" --file large.txt \
  --root-model openai/gpt-4.1 \
  --worker-model openai/gpt-4.1-mini

# From stdin
cat document.txt | python main.py "Extract key points" --stdin
```

### Python API

```python
import asyncio
from rlm import RLM

async def main():
    rlm = RLM(
        root_model="openai/gpt-4.1",      # Smart model for planning
        worker_model="openai/gpt-4.1-mini" # Cheap model for sub_llm
    )

    context = open("large_file.txt").read()  # 100K+ chars
    answer = await rlm.query("Find all errors", context, verbose=True)
    print(answer)

asyncio.run(main())
```

## Features

### Tools Available to LLM

| Tool | Description |
|------|-------------|
| `peek(start, end)` | View character range |
| `peek_lines(start, end)` | View line range |
| `grep(pattern)` | Regex search, returns matches with line numbers |
| `chunk(chunk_size)` | Get chunk boundaries for splitting |
| `sub_llm(query, start, end)` | **Recursive call on context slice (parallel!)** |
| `exec_python(code)` | Execute Python with access to context |
| `set_variable(name, value)` | Store intermediate result |
| `get_variable(name)` | Retrieve stored result |
| `final(answer)` | Return final answer |
| `final_var(name)` | Return variable as answer |

### Verbose Output

```
Context: 414,935 chars, 7,043 lines
Root model: openai/gpt-4.1
Worker model: openai/gpt-4.1-mini
--------------------------------------------------
[async] iter 1: calling gpt-4.1 (root)...
  [sync] chunk({"chunk_size": 80000})
[async] iter 2: calling gpt-4.1 (root)...
  [async] sub_llm(...) x6 parallel
    [async] sub_llm started (depth=1, 80,000 chars) [gpt-4.1-mini]
    [async] iter 1: calling gpt-4.1-mini (worker)...

┌─ Execution Trace ─────────────────────────────
│ │ [async] query(depth=0, 414,935 chars)
│    ├─ [async] LLM call #1 (1200ms)
│    │  └─ [sync] chunk(...) (0ms)
│    ├─ [async] LLM call #2 (800ms)
│    │  ├─ [parallel x6]
│    │  │  ├─ [async] sub_llm(...) → ...
│    │  │  └─ [async] sub_llm(...) → ...
└───────────────────────────────────────────────

══════════════════════════════════════════════════
  LLM calls: 15  │  Iterations: 15  │  Tools: 20
  sub_llm calls: 6  │  Max depth: 2
  Tokens: 45,000 in / 1,200 out
  Time: 12.34s
  Tools: chunk:1, final:1, grep:8, sub_llm:6
══════════════════════════════════════════════════
```

## Architecture

```
rlm/
├── rlm/
│   ├── __init__.py      # Exports RLM, run_sync
│   ├── client.py        # OpenRouter async client
│   ├── sandbox.py       # REPL environment (context storage)
│   ├── tools.py         # Tool definitions & execution
│   ├── prompts.py       # System prompts
│   └── rlm.py           # Main orchestrator
├── main.py              # CLI entry point
├── requirements.txt
└── .env                 # API key
```

## Key Design Decisions

- **Dual-model architecture**: Smart root model plans, cheap workers execute
- **Async parallel sub_llm**: Multiple chunks processed simultaneously
- **Full Python exec**: Workers can run arbitrary code for complex processing
- **Tool-based approach**: Uses OpenAI function calling for reliable tool use

## References

- [Recursive Language Models (arXiv)](https://arxiv.org/abs/2512.24601)
- [RLM Blog Post - Alex Zhang](https://alexzhang13.github.io/blog/2025/rlm/)
- [Prime Intellect: RLM Paradigm of 2026](https://www.primeintellect.ai/blog/rlm)
