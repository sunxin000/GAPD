from types import SimpleNamespace
import unittest

from kbqa_r1.gapd.state_matcher import jaccard_similarity, match_state_to_gold_step
from kbqa_r1.utils.reward_score.mid_reward import calculate_mid_prf1, extract_mid_list
from kbqa_custom_reward import compute_score
from tools.check_gold_trace import is_acceptable, trace_coverage


class StateMatchingTests(unittest.TestCase):
    def test_empty_and_disjoint_states(self):
        self.assertEqual(jaccard_similarity(set(), set()), 1.0)
        self.assertEqual(jaccard_similarity(set(), {"m.a"}), 0.0)
        self.assertIsNone(match_state_to_gold_step({"m.a"}, []))
        self.assertIsNone(match_state_to_gold_step({"m.a"}, [SimpleNamespace(prev_mid_set={"m.b"})]))

    def test_jaccard_threshold_boundary(self):
        full = {f"m.{i}" for i in range(20)}
        trace = [SimpleNamespace(prev_mid_set=full)]
        self.assertEqual(match_state_to_gold_step(full - {"m.0"}, trace), 0)
        self.assertIsNone(match_state_to_gold_step(full - {"m.0", "m.1"}, trace))

    def test_best_match_and_stable_ties(self):
        trace = [SimpleNamespace(prev_mid_set={"m.a", "m.b"}),
                 SimpleNamespace(prev_mid_set={"m.a"}), SimpleNamespace(prev_mid_set={"m.a"})]
        self.assertEqual(match_state_to_gold_step({"m.a"}, trace), 1)


class RewardTests(unittest.TestCase):
    def test_set_f1_and_duplicate_answers(self):
        precision, recall, f1 = calculate_mid_prf1(["m.a", "m.b"], ["m.a", "m.a"])
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_final_answer_takes_precedence(self):
        text = '<answer>m.wrong</answer><answer>["m.a", "m.b"]</answer>'
        self.assertEqual(extract_mid_list(text), ["m.a", "m.b"])
        score = compute_score("webqsp", text, {"target": ["m.a", "m.b"]})
        self.assertEqual(score["mid_f1"], 1.0)
        self.assertEqual(score["score"], 1.0)

    def test_no_answer_receives_zero(self):
        score = compute_score("grailqa", "<think>unfinished</think>", ["m.a"])
        self.assertEqual(score["score"], 0.0)


class TraceValidationTests(unittest.TestCase):
    def row(self, next_mids=None):
        return {"extra_info": {"gold_trace": [{"prev_mid_set": [],
                "next_mid_set": next_mids if next_mids is not None else ["m.a"],
                "per_turn_hint": "Find_relation [ m.a | example.relation ]"}]}}

    def test_valid_initial_state_and_coverage_boundary(self):
        stats = trace_coverage([self.row() for _ in range(9)] + [{}])
        self.assertTrue(is_acceptable(stats))
        self.assertFalse(is_acceptable(trace_coverage([self.row() for _ in range(8)] + [{}, {}])))

    def test_empty_missing_and_malformed_traces_fail(self):
        for rows in ([], [{}], [self.row([])], [{"extra_info": {"gold_trace": ["bad-step"]}}]):
            self.assertFalse(is_acceptable(trace_coverage(rows)))

    def test_array_backed_parquet_fields(self):
        # Test the Arrow/NumPy normalization protocol without requiring NumPy.
        class ArrayLike:
            def __init__(self, value): self.value = value
            def tolist(self): return self.value
        row = self.row()
        row["extra_info"]["gold_trace"][0]["next_mid_set"] = ArrayLike(["m.a"])
        row["extra_info"]["gold_trace"] = ArrayLike(row["extra_info"]["gold_trace"])
        self.assertTrue(is_acceptable(trace_coverage([row])))


if __name__ == "__main__":
    unittest.main()
