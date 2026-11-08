# context-packer

Deterministic token-budget packing for agent context windows. Offline,
standard library only, no model calls.

LLM agents accumulate system prompts, user turns, assistant replies and tool
results until the context window overflows. `context_packer.pack()` decides
what to keep, deterministically:

1. Pinned and system messages always survive (truncated only if one alone
   exceeds the budget).
2. The most recent turns are preferred.
3. Oversized tool results are truncated to a configurable cap first.
4. Same input always produces the same output - testable, replayable.

```python
from context_packer import Message, pack

messages = [
    Message("system", "You are a careful assistant.", pinned=True),
    Message("user", "Summarize these logs."),
    Message("tool", "... 50k tokens of logs ..."),
    Message("user", "And the errors?"),
]
packed, report = pack(messages, budget=8000, keep_recent=2, max_tool_tokens=500)
print(report.used, report.dropped, report.truncated)
```

## Honest scope

- Token counts are a heuristic (~4 chars/token + per-message overhead).
  Calibrate `CHARS_PER_TOKEN` against your real tokenizer for tight margins.
- Truncation is a plain character cut with a marker, not a model-written
  summary. Wiring an LLM summarizer is a deliberate, separate step.
- This manages *what fits*, not *what matters semantically*; importance is
  recency + pins, not learned relevance.

## Tests

```
python3 -m unittest discover -s tests
```

## Demo

`demo.py` reproduces the recorded demo run end to end, offline (standard
library only, no model calls):

    python demo.py

It executes the library's real code path and prints the same output shown in
the recorded demo video.
