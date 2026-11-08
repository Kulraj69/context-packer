"""Reproduces the demo video's real output: pack 28 messages into 500 tokens."""
from context_packer import Message, pack

msgs = [Message(role="system", content="You are a travel booking agent.", pinned=True),
        Message(role="user", content="Book the cheapest BLR-DEL flight tomorrow morning", pinned=True)]
for i in range(12):
    msgs.append(Message(role="tool", content="search result batch %d: " % i + "offer details " * 18))
    msgs.append(Message(role="assistant", content="checking options in batch %d" % i))
msgs.append(Message(role="user", content="window seat if possible"))
msgs.append(Message(role="assistant", content="Noted: window seat."))
if __name__ == "__main__":
    packed, report = pack(msgs, 500)
    print(f"input: {len(msgs)} messages")
    print(f"packed: kept {report.kept}, dropped {report.dropped}, truncated {report.truncated}")
    print(f"budget: used {report.used}/{report.budget} tokens")
    print("dropped oldest first:", ", ".join(report.dropped_roles))
