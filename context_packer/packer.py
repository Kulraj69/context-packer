"""Deterministic token-budget packing for agent context windows.

Agents accumulate system prompts, user turns, assistant replies and tool
results. Models have a fixed context window, so something has to give.
This module decides *what* to keep, deterministically: same input, same
output, no model call required.

Heuristic token estimate: ~4 characters per token plus a small per-message
overhead. Calibrate CHARS_PER_TOKEN against your real tokenizer if the
margin matters; the packing logic is independent of the estimate.
"""

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 4
PER_MESSAGE_OVERHEAD = 4  # role markers, separators


def estimate_tokens(text: str) -> int:
    """Rough token count for a string. Deterministic upper-biased estimate."""
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


@dataclass
class Message:
    """One context entry. Pinned messages are never dropped or truncated."""
    role: str            # 'system' | 'user' | 'assistant' | 'tool'
    content: str
    pinned: bool = False

    def tokens(self) -> int:
        return PER_MESSAGE_OVERHEAD + estimate_tokens(self.content)

    def truncated(self, max_tokens: int) -> "Message":
        """Return a copy whose content fits max_tokens (including overhead)."""
        budget_chars = max(0, (max_tokens - PER_MESSAGE_OVERHEAD) * CHARS_PER_TOKEN)
        if len(self.content) <= budget_chars:
            return Message(self.role, self.content, self.pinned)
        marker = " [...truncated]"
        keep = max(0, budget_chars - len(marker))
        return Message(self.role, self.content[:keep] + marker, self.pinned)


@dataclass
class PackReport:
    budget: int
    used: int
    kept: int
    dropped: int
    truncated: int
    dropped_roles: list = field(default_factory=list)


def pack(messages, budget: int, keep_recent: int = 2, max_tool_tokens: int = 200):
    """Pack messages into a token budget, deterministically.

    Order of preference:
      1. Pinned and system messages (truncated if alone over budget).
      2. The `keep_recent` most recent non-system messages.
      3. Older messages, most recent first, while budget remains.
    Tool messages are truncated to `max_tool_tokens` before packing.

    Returns (packed_messages_in_original_order, PackReport).
    """
    if budget <= 0:
        raise ValueError("budget must be positive")

    # Pre-truncate oversized tool results.
    prepared = []
    trunc_count = 0
    for m in messages:
        if m.role == "tool" and not m.pinned and m.tokens() > max_tool_tokens:
            prepared.append(m.truncated(max_tool_tokens))
            trunc_count += 1
        else:
            prepared.append(m)

    n = len(prepared)
    system_idx = [i for i, m in enumerate(prepared) if m.role == "system" or m.pinned]
    other_idx = [i for i in range(n) if i not in system_idx]

    keep = set()
    used = 0

    # 1. Pinned / system first.
    for i in system_idx:
        m = prepared[i]
        if m.tokens() <= budget - used:
            keep.add(i)
            used += m.tokens()
        else:
            # System prompt alone exceeds remaining budget: truncate to fit.
            fitted = m.truncated(max(1, budget - used))
            prepared[i] = fitted
            keep.add(i)
            used += fitted.tokens()
            trunc_count += 1
            break  # budget is now exhausted

    # 2. Most recent non-system messages.
    recent = other_idx[-keep_recent:] if keep_recent > 0 else []
    for i in recent:
        m = prepared[i]
        if i not in keep and m.tokens() <= budget - used:
            keep.add(i)
            used += m.tokens()

    # 3. Older messages, most recent first.
    for i in reversed(other_idx):
        if i in keep:
            continue
        m = prepared[i]
        if m.tokens() <= budget - used:
            keep.add(i)
            used += m.tokens()

    packed = [prepared[i] for i in range(n) if i in keep]
    report = PackReport(
        budget=budget,
        used=used,
        kept=len(keep),
        dropped=n - len(keep),
        truncated=trunc_count,
        dropped_roles=[prepared[i].role for i in range(n) if i not in keep],
    )
    return packed, report
