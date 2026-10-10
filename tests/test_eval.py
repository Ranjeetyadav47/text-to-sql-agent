"""The eval harness itself: the comparison semantics it is built on, and the numbers it
reports. If `results_match` is wrong, every other number in this repo is decoration."""

import os
import unittest

from t2s.eval import results_match, run_eval, write_report
from t2s.data import build_questions

from helpers import Workspace


class ResultComparisonTests(unittest.TestCase):
    """Execution accuracy lives or dies on these. Each one is a way a naive comparison lies."""

    def test_row_order_is_ignored_unless_gold_ordered_them(self):
        a, b = [(1, "x"), (2, "y")], [(2, "y"), (1, "x")]
        self.assertTrue(results_match(a, b, order_matters=False))
        self.assertFalse(results_match(a, b, order_matters=True))

    def test_duplicate_rows_are_not_silently_forgiven(self):
        """A set-based comparison would call these equal. They are not."""
        self.assertFalse(results_match([(1,), (1,)], [(1,)], order_matters=False))

    def test_floats_are_compared_to_two_decimal_places(self):
        self.assertTrue(results_match([(1.234567,)], [(1.23,)], order_matters=False))
        self.assertFalse(results_match([(1.24,)], [(1.23,)], order_matters=False))

    def test_missing_and_extra_rows_both_fail(self):
        self.assertFalse(results_match([(1,)], [(1,), (2,)], order_matters=False))
        self.assertFalse(results_match([(1,), (2,)], [(1,)], order_matters=False))

    def test_empty_results_match_each_other(self):
        self.assertTrue(results_match([], [], order_matters=False))

    def test_nulls_compare_equal(self):
        self.assertTrue(results_match([(1, None)], [(1, None)], order_matters=False))


class QuestionSetTests(unittest.TestCase):
    def test_the_set_is_90_questions_with_the_documented_mix(self):
        qs = build_questions()
        self.assertEqual(len(qs), 90)
        counts = {}
        for q in qs:
            counts[q["category"]] = counts.get(q["category"], 0) + 1
        self.assertEqual(counts, {
            "simple_filter": 12, "count": 12, "aggregate": 12, "join": 12,
            "group_topk": 12, "date_range": 10, "negation": 10, "ambiguous": 10,
        })

    def test_half_the_set_is_paraphrases(self):
        qs = build_questions()
        self.assertEqual(sum(1 for q in qs if q["is_paraphrase"]), 45)

    def test_only_ambiguous_questions_expect_clarification_and_have_no_gold(self):
        for q in build_questions():
            with self.subTest(id=q["id"]):
                if q["category"] == "ambiguous":
                    self.assertEqual(q["expected"], "clarify")
                    self.assertIsNone(q["gold_sql"])
                else:
                    self.assertEqual(q["expected"], "sql")
                    self.assertTrue(q["gold_sql"])

    def test_question_ids_are_unique(self):
        ids = [q["id"] for q in build_questions()]
        self.assertEqual(len(ids), len(set(ids)))


class EvalRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace(seed=7)
        cls.result = run_eval(cls.ws.data_dir, cls.ws.db_path, max_tables=4, label="test")

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def test_scores_the_whole_set(self):
        self.assertEqual(self.result.n, 90)
        self.assertEqual(self.result.n_sql, 80)
        self.assertEqual(self.result.n_clarify, 10)

    def test_ci_gate_template_accuracy_holds(self):
        self.assertGreaterEqual(self.result.template_accuracy, 0.95)

    def test_ci_gate_no_unsafe_query_ever_executes(self):
        self.assertEqual(self.result.unsafe_queries, 0)
        self.assertEqual(self.result.attacks_blocked, self.result.attacks_total)

    def test_mock_accuracy_sits_in_the_band_the_readme_claims(self):
        """The README says 80-90% and explains why it is deliberately not higher. If the
        mock drifts out of that band, the README is wrong and this fails first."""
        self.assertGreater(self.result.execution_accuracy, 0.80)
        self.assertLess(self.result.execution_accuracy, 0.90)

    def test_paraphrases_score_worse_than_templates(self):
        """The honest signal: the gap is what says the score is pattern-matching, not
        understanding. If it ever closes, the paraphrases have leaked into the patterns."""
        self.assertLess(self.result.paraphrase_accuracy, self.result.template_accuracy)

    def test_vague_questions_are_asked_about_not_answered(self):
        self.assertEqual(self.result.clarification_accuracy, 1.0)

    def test_failures_are_concentrated_in_paraphrases_and_negation(self):
        self.assertTrue(self.result.failures)
        paraphrase_failures = sum(1 for f in self.result.failures if f.is_paraphrase)
        self.assertGreater(paraphrase_failures, len(self.result.failures) / 2)

    def test_no_gold_query_is_broken(self):
        """A gold_error means the labelled set is wrong, which invalidates every number."""
        broken = [f for f in self.result.failures if f.reason == "gold_error"]
        self.assertEqual(broken, [], f"broken gold queries: {broken}")

    def test_schema_linking_cuts_prompt_tokens(self):
        unlinked = run_eval(self.ws.data_dir, self.ws.db_path, max_tables=None, label="whole")
        self.assertLess(self.result.prompt_tokens, unlinked.prompt_tokens)

    def test_report_is_written_with_the_sections_that_matter(self):
        path = write_report([self.result], os.path.join(self.ws.dir, "reports", "r.md"))
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for section in ("Execution accuracy", "on paraphrases", "by category",
                        "where it breaks", "Known attacks blocked"):
            self.assertIn(section, text)


if __name__ == "__main__":
    unittest.main()
