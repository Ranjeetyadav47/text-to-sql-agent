"""The agent loop end to end: every category's template form, the clarify path, the
correction round, the guard path, and the executor's time limit."""

import json
import unittest

from t2s.config import GuardConfig, LinkConfig
from t2s.executor import Executor
from t2s.generator import _parse
from t2s.llm.base import LLMResponse
from t2s.models import AnswerStatus, PermanentError
from t2s.llm.mock import MockLLM

from helpers import Workspace, gold_rows


class StubLLM:
    """Returns a scripted sequence of payloads, so a specific failure can be staged."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.tasks = []

    def complete(self, task, system, user, json_mode=False):
        self.tasks.append(task)
        payload = self.payloads.pop(0) if self.payloads else {"sql": None}
        body = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(content=body, prompt_tokens=10, completion_tokens=5, model="stub")


class AgentHappyPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace(seed=7)
        cls.agent = cls.ws.agent(guard=GuardConfig(row_limit=5000))

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def test_mock_handles_every_categorys_template_form(self):
        """One representative template per category. These are the questions the CI gate
        is really protecting; if one breaks, the category name says which."""
        cases = {
            "simple_filter": ("customers in Pune",
                              "SELECT customer_id, name, city FROM customers WHERE city = 'Pune'"),
            "count": ("how many tickets are still open",
                      "SELECT COUNT(*) AS n FROM service_tickets WHERE status = 'open'"),
            "aggregate": ("average unit cost of parts",
                          "SELECT ROUND(AVG(unit_cost), 2) AS avg_unit_cost FROM parts"),
            "join": ("tickets for equipment model Helios 400",
                     "SELECT st.ticket_id, e.model FROM service_tickets st "
                     "JOIN equipment e ON st.equipment_id = e.equipment_id "
                     "WHERE e.model = 'Helios 400'"),
            "group_topk": ("top 5 fault codes by ticket count",
                           "SELECT fault_code, COUNT(*) AS n FROM service_tickets "
                           "GROUP BY fault_code ORDER BY n DESC, fault_code ASC LIMIT 5"),
            "date_range": ("tickets closed in 2025",
                           "SELECT ticket_id, closed_at FROM service_tickets "
                           "WHERE closed_at >= '2025-01-01' AND closed_at < '2026-01-01'"),
            "negation": ("parts never used on any ticket",
                         "SELECT p.part_id, p.name FROM parts p WHERE NOT EXISTS "
                         "(SELECT 1 FROM ticket_parts tp WHERE tp.part_id = p.part_id)"),
        }
        for category, (question, gold) in cases.items():
            with self.subTest(category=category):
                answer = self.agent.ask(question)
                self.assertEqual(answer.status, AnswerStatus.OK, answer.error or answer.violations)
                self.assertEqual(sorted(map(repr, answer.rows)),
                                 sorted(map(repr, gold_rows(self.ws, gold))))

    def test_answer_always_carries_the_sql_and_the_linked_tables(self):
        answer = self.agent.ask("top 5 fault codes by ticket count")
        self.assertTrue(answer.sql)
        self.assertIn("service_tickets", answer.linked_tables)
        self.assertTrue(answer.summary)

    def test_ambiguous_question_asks_instead_of_guessing(self):
        answer = self.agent.ask("which ones are bad")
        self.assertEqual(answer.status, AnswerStatus.CLARIFY)
        self.assertIsNone(answer.sql)
        self.assertTrue(answer.clarifying_question)

    def test_unmatched_but_concrete_question_guesses_rather_than_asking(self):
        """The mock's realistic failure mode: a confident, plausible, wrong query. If this
        ever starts asking instead, the eval stops measuring the failure that matters."""
        answer = self.agent.ask("where are our Pune customers")
        self.assertEqual(answer.status, AnswerStatus.OK)
        self.assertIn("SELECT *", answer.sql)

    def test_row_cap_is_applied_when_the_query_has_no_limit(self):
        capped = self.ws.agent(guard=GuardConfig(row_limit=5))
        answer = capped.ask("tickets closed in 2025")
        self.assertEqual(answer.status, AnswerStatus.OK)
        self.assertLessEqual(len(answer.rows), 5)
        self.assertIn("LIMIT 5", answer.sql)


class AgentFailurePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace(seed=7)

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def test_correction_round_fixes_a_bad_column_and_succeeds(self):
        llm = StubLLM(
            {"sql": "SELECT customer_name FROM customers", "confidence": 0.9},
            {"sql": "SELECT name FROM customers", "confidence": 0.7},
        )
        answer = self.ws.agent(llm=llm).ask("customers in Pune")
        self.assertEqual(answer.status, AnswerStatus.OK, answer.error)
        self.assertTrue(answer.corrected)
        self.assertEqual(answer.attempts, 2)
        self.assertEqual(llm.tasks, ["generate", "correct"])

    def test_the_real_mock_can_repair_its_own_bad_column(self):
        llm = MockLLM()
        fixed = llm._correct("customers in Pune", "SELECT customer_name FROM customers",
                             "no such column: customer_name", "customers")
        self.assertEqual(fixed["sql"], "SELECT name FROM customers")

    def test_a_second_failure_returns_structured_failure_not_an_exception(self):
        llm = StubLLM(
            {"sql": "SELECT nope FROM customers", "confidence": 0.9},
            {"sql": "SELECT still_nope FROM customers", "confidence": 0.5},
        )
        answer = self.ws.agent(llm=llm).ask("customers in Pune")
        self.assertEqual(answer.status, AnswerStatus.FAILED)
        self.assertEqual(answer.attempts, 2)
        self.assertIsNotNone(answer.error)

    def test_only_one_correction_round_is_ever_attempted(self):
        llm = StubLLM({"sql": "SELECT nope FROM customers"}, {"sql": "SELECT nah FROM customers"},
                      {"sql": "SELECT nope2 FROM customers"})
        self.ws.agent(llm=llm).ask("customers in Pune")
        self.assertEqual(llm.tasks, ["generate", "correct"])

    def test_unsafe_generated_sql_is_blocked_before_execution(self):
        llm = StubLLM({"sql": "DROP TABLE customers", "confidence": 0.99})
        answer = self.ws.agent(llm=llm).ask("customers in Pune")
        self.assertEqual(answer.status, AnswerStatus.BLOCKED)
        self.assertTrue(answer.violations)
        self.assertEqual(llm.tasks, ["generate"], "a blocked query must not trigger a correction")

    def test_non_json_model_output_is_a_permanent_error(self):
        llm = StubLLM("Sure! Here is your SQL: SELECT 1")
        with self.assertRaises(PermanentError):
            self.ws.agent(llm=llm).ask("customers in Pune")

    def test_fenced_json_is_tolerated(self):
        parsed = _parse('```json\n{"sql": "SELECT 1", "confidence": 0.5}\n```')
        self.assertEqual(parsed.sql, "SELECT 1")

    def test_model_returning_null_sql_without_a_question_fails_cleanly(self):
        llm = StubLLM({"sql": None, "needs_clarification": False})
        answer = self.ws.agent(llm=llm).ask("customers in Pune")
        self.assertEqual(answer.status, AnswerStatus.FAILED)


class ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace(seed=7)

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def test_times_out_on_a_deliberately_slow_query(self):
        """A cartesian join over the ticket table, bounded only by the clock."""
        slow = ("SELECT COUNT(*) FROM service_tickets a, service_tickets b, "
                "service_tickets c, service_tickets d")
        result = Executor(self.ws.db_path, GuardConfig(timeout_s=0.25)).run(slow)
        self.assertFalse(result.ok)
        self.assertTrue(result.timed_out)
        self.assertIn("time limit", result.error)

    def test_the_connection_is_read_only(self):
        result = Executor(self.ws.db_path).run("DELETE FROM parts")
        self.assertFalse(result.ok)
        self.assertFalse(result.timed_out, "a write rejection is not a timeout")

    def test_a_bad_column_reports_the_sqlite_error_for_the_correction_round(self):
        result = Executor(self.ws.db_path).run("SELECT nope FROM parts")
        self.assertFalse(result.ok)
        self.assertIn("no such column", result.error)

    def test_successful_run_returns_columns_and_rows(self):
        result = Executor(self.ws.db_path).run("SELECT part_id, name FROM parts LIMIT 3")
        self.assertTrue(result.ok)
        self.assertEqual(result.columns, ["part_id", "name"])
        self.assertEqual(len(result.rows), 3)


if __name__ == "__main__":
    unittest.main()
