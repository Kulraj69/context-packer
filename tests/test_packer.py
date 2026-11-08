import unittest

from context_packer import Message, estimate_tokens, pack


def msg(role, words=10, pinned=False):
    return Message(role, " ".join(["word"] * words), pinned)


class TestEstimateTokens(unittest.TestCase):
    def test_empty_is_one(self):
        self.assertEqual(estimate_tokens(""), 1)

    def test_scales_with_length(self):
        self.assertGreater(estimate_tokens("abcd" * 100), estimate_tokens("abcd"))

    def test_deterministic(self):
        self.assertEqual(estimate_tokens("hello world"), estimate_tokens("hello world"))


class TestPack(unittest.TestCase):
    def test_fits_budget_exactly(self):
        msgs = [msg("user", 20) for _ in range(10)]
        packed, report = pack(msgs, budget=100, keep_recent=0)
        self.assertLessEqual(report.used, 100)
        self.assertEqual(report.used, sum(m.tokens() for m in packed))

    def test_system_always_kept(self):
        msgs = [msg("system", 5)] + [msg("user", 50) for _ in range(10)]
        packed, _ = pack(msgs, budget=60, keep_recent=0)
        self.assertEqual(packed[0].role, "system")

    def test_pinned_never_dropped(self):
        msgs = [msg("user", 5)] * 5 + [msg("tool", 5, pinned=True)] + [msg("user", 50)] * 5
        packed, _ = pack(msgs, budget=80, keep_recent=0)
        self.assertTrue(any(m.pinned for m in packed))

    def test_recency_preference(self):
        msgs = [msg("user", 10) for _ in range(6)]
        packed, _ = pack(msgs, budget=40, keep_recent=2)
        # The survivors must be a suffix of the originals (most recent kept).
        self.assertEqual(packed, msgs[-len(packed):])

    def test_oversized_tool_truncated(self):
        big_tool = msg("tool", 2000)
        packed, report = pack([big_tool], budget=500, max_tool_tokens=100)
        self.assertLessEqual(packed[0].tokens(), 100)
        self.assertTrue(packed[0].content.endswith("[...truncated]"))
        self.assertEqual(report.truncated, 1)

    def test_drop_count_consistent(self):
        msgs = [msg("user", 10) for _ in range(10)]
        packed, report = pack(msgs, budget=50, keep_recent=0)
        self.assertEqual(report.kept + report.dropped, 10)
        self.assertEqual(report.dropped, len(report.dropped_roles))

    def test_determinism(self):
        msgs = [msg("system", 5)] + [msg("user", 30) for _ in range(8)]
        a, _ = pack(msgs, budget=200)
        b, _ = pack(msgs, budget=200)
        self.assertEqual([m.content for m in a], [m.content for m in b])

    def test_zero_budget_raises(self):
        with self.assertRaises(ValueError):
            pack([msg("user")], budget=0)

    def test_empty_input(self):
        packed, report = pack([], budget=100)
        self.assertEqual(packed, [])
        self.assertEqual(report.used, 0)

    def test_oversized_system_truncated_not_dropped(self):
        msgs = [msg("system", 500)]
        packed, report = pack(msgs, budget=50)
        self.assertEqual(len(packed), 1)
        self.assertLessEqual(packed[0].tokens(), 50)

    def test_preserves_original_order(self):
        msgs = [msg("system", 5), msg("user", 10), msg("assistant", 10), msg("user", 10)]
        packed, _ = pack(msgs, budget=1000)
        self.assertEqual([m.role for m in packed], ["system", "user", "assistant", "user"])


if __name__ == "__main__":
    unittest.main()
