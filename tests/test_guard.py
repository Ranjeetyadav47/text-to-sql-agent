"""The guard is the security boundary, so it is tested against attacks, not happy paths.

Every statement in `KNOWN_ATTACKS` gets its own assertion via subTest, so a regression names
the exact attack that got through rather than saying "18 attacks, one failed".
"""

import unittest

from t2s.config import GuardConfig
from t2s.guard import KNOWN_ATTACKS, Guard, mask_literals, strip_comments

TABLES = ["customers", "equipment", "technicians", "service_tickets", "parts", "ticket_parts"]


class GuardBlocksAttacksTests(unittest.TestCase):
    def setUp(self):
        self.guard = Guard(TABLES)

    def test_every_known_attack_is_blocked(self):
        self.assertGreaterEqual(len(KNOWN_ATTACKS), 8, "the attack corpus is the point of this suite")
        for attack in KNOWN_ATTACKS:
            with self.subTest(attack=attack):
                verdict = self.guard.check(attack)
                self.assertFalse(verdict.allowed, f"guard allowed: {attack}")
                self.assertTrue(verdict.violations)

    def test_write_statements_are_rejected(self):
        for sql in ("INSERT INTO parts VALUES (1,'x',1.0)", "UPDATE parts SET unit_cost = 0",
                    "DELETE FROM parts", "DROP TABLE parts", "ALTER TABLE parts ADD COLUMN x TEXT"):
            with self.subTest(sql=sql):
                self.assertFalse(self.guard.check(sql).allowed)

    def test_statement_chaining_is_rejected_even_after_a_valid_select(self):
        v = self.guard.check("SELECT 1 FROM parts; DROP TABLE parts")
        self.assertFalse(v.allowed)
        self.assertIn("multiple statements are not allowed", v.violations)

    def test_comments_are_rejected_because_they_hide_payloads(self):
        self.assertFalse(self.guard.check("SELECT * FROM parts -- ; DROP TABLE parts").allowed)
        self.assertFalse(self.guard.check("SELECT * FROM parts /* x */ WHERE 1=1").allowed)

    def test_sqlite_internals_are_unreachable(self):
        for sql in ("SELECT name FROM sqlite_master",
                    "SELECT part_id FROM parts UNION SELECT name FROM sqlite_master",
                    "SELECT sql FROM sqlite_schema"):
            with self.subTest(sql=sql):
                self.assertFalse(self.guard.check(sql).allowed)

    def test_unknown_tables_are_rejected(self):
        v = self.guard.check("SELECT * FROM secret_audit_log")
        self.assertFalse(v.allowed)
        self.assertIn("unknown table: secret_audit_log", v.violations)

    def test_empty_and_oversized_statements_are_rejected(self):
        self.assertFalse(self.guard.check("").allowed)
        self.assertFalse(self.guard.check(None).allowed)
        giant = "SELECT * FROM parts WHERE name = '" + "x" * 5000 + "'"
        self.assertFalse(Guard(TABLES, GuardConfig(max_sql_chars=4000)).check(giant).allowed)


class GuardAllowsLegitimateQueriesTests(unittest.TestCase):
    def setUp(self):
        self.guard = Guard(TABLES)

    def test_plain_select_passes_and_gets_a_limit(self):
        v = self.guard.check("SELECT customer_id FROM customers")
        self.assertTrue(v.allowed)
        self.assertTrue(v.sql.endswith("LIMIT 200"))
        self.assertEqual(v.rewrites, ["appended LIMIT 200"])

    def test_existing_limit_is_left_alone(self):
        v = self.guard.check("SELECT customer_id FROM customers LIMIT 5")
        self.assertTrue(v.allowed)
        self.assertEqual(v.sql, "SELECT customer_id FROM customers LIMIT 5")
        self.assertEqual(v.rewrites, [])

    def test_cte_names_are_not_mistaken_for_unknown_tables(self):
        v = self.guard.check("WITH recent AS (SELECT * FROM service_tickets) SELECT * FROM recent")
        self.assertTrue(v.allowed, v.violations)

    def test_joins_and_aliases_resolve_to_real_tables(self):
        v = self.guard.check("SELECT c.name FROM customers c JOIN equipment e "
                             "ON c.customer_id = e.customer_id")
        self.assertTrue(v.allowed, v.violations)

    def test_a_trailing_semicolon_is_tolerated(self):
        self.assertTrue(self.guard.check("SELECT 1 FROM parts;").allowed)

    def test_subqueries_are_not_treated_as_table_names(self):
        v = self.guard.check("SELECT * FROM (SELECT part_id FROM parts) t")
        self.assertTrue(v.allowed, v.violations)


class LiteralMaskingTests(unittest.TestCase):
    """The masking is why a keyword inside a value is not an attack, and why an attack
    outside a value is not hidden by one. Both directions are tested."""

    def test_keywords_inside_string_literals_are_not_attacks(self):
        v = Guard(TABLES).check("SELECT * FROM customers WHERE name = 'Delete Logistics'")
        self.assertTrue(v.allowed, v.violations)

    def test_semicolon_inside_a_literal_is_not_statement_chaining(self):
        v = Guard(TABLES).check("SELECT * FROM customers WHERE name = 'a; DROP TABLE parts'")
        self.assertTrue(v.allowed, v.violations)

    def test_escaped_quotes_do_not_break_out_of_the_literal(self):
        v = Guard(TABLES).check("SELECT * FROM customers WHERE name = 'it''s fine'")
        self.assertTrue(v.allowed, v.violations)

    def test_masking_preserves_length_so_offsets_stay_valid(self):
        sql = "SELECT * FROM t WHERE a = 'hello' AND b = 'x'"
        self.assertEqual(len(mask_literals(sql)), len(sql))

    def test_comment_stripping_reports_whether_it_found_any(self):
        _, found = strip_comments("SELECT 1 -- hi")
        self.assertTrue(found)
        _, found = strip_comments("SELECT 1")
        self.assertFalse(found)


if __name__ == "__main__":
    unittest.main()
