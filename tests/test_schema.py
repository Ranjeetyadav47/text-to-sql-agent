"""Schema loading and linking: does it pick the right few tables, and does it know when
it cannot tell? The second question matters more, because an empty link is what turns into
a clarifying question instead of a guess."""

import unittest

from t2s.config import LinkConfig
from t2s.schema import SchemaLinker, tokenise

from helpers import Workspace


class SchemaLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace(seed=3)

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def test_all_six_tables_load_from_sqlite(self):
        self.assertEqual(sorted(self.ws.table_names()),
                         ["customers", "equipment", "parts", "service_tickets",
                          "technicians", "ticket_parts"])

    def test_foreign_keys_build_a_connected_graph(self):
        s = self.ws.schema
        self.assertIn("equipment", s.neighbours("customers"))
        self.assertIn("service_tickets", s.neighbours("equipment"))
        self.assertIn("parts", s.neighbours("ticket_parts"))
        # customers and service_tickets are two hops apart, not adjacent.
        self.assertNotIn("service_tickets", s.neighbours("customers"))

    def test_join_hint_names_the_actual_key(self):
        hint = self.ws.schema.join_hint("equipment", "customers")
        self.assertEqual(hint, "equipment.customer_id = customers.customer_id")
        self.assertIsNone(self.ws.schema.join_hint("customers", "parts"))

    def test_columns_carry_descriptions_and_keys(self):
        tickets = self.ws.schema.by_name["service_tickets"]
        self.assertTrue(tickets.column("ticket_id").is_pk)
        self.assertEqual(tickets.column("equipment_id").references, ("equipment", "equipment_id"))
        self.assertIn("urgency", tickets.column("priority").description)

    def test_render_includes_only_the_requested_tables(self):
        text = self.ws.schema.render(["parts"])
        self.assertIn("TABLE parts", text)
        self.assertNotIn("TABLE customers", text)


class SchemaLinkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace(seed=3)
        cls.linker = SchemaLinker(cls.ws.schema, LinkConfig(max_tables=4))

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def test_picks_the_right_tables_for_known_questions(self):
        cases = [
            ("customers in Pune", "customers"),
            ("parts that cost more than 5000", "parts"),
            ("technicians in the West region", "technicians"),
            ("tickets with fault code F-101", "service_tickets"),
            ("total quantity of parts used", "ticket_parts"),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertIn(expected, self.linker.link(question).names)

    def test_never_exceeds_the_table_cap(self):
        wordy = ("technicians who closed tickets for customers in Pune using parts on equipment "
                 "with a high priority fault code in the West region")
        self.assertLessEqual(len(self.linker.link(wordy).names), 4)

    def test_a_question_naming_nothing_links_nothing(self):
        """This is what becomes a clarifying question rather than a guess."""
        for question in ("which ones are bad", "the usual report", "how is it going"):
            with self.subTest(question=question):
                self.assertEqual(self.linker.link(question).names, [])

    def test_every_chosen_table_gets_a_reason(self):
        linked = self.linker.link("technicians who closed tickets for customers in Pune")
        self.assertTrue(linked.names)
        for name in linked.names:
            self.assertIn(name, linked.reasons)
        self.assertEqual(linked.considered, 6)

    def test_synonyms_bridge_user_words_to_schema_words(self):
        """The schema never says "machine", but users do."""
        self.assertIn("equipment", self.linker.link("machines that never needed service").names)

    def test_tokenise_drops_stopwords_and_keeps_underscores(self):
        self.assertEqual(tokenise("how many of the mid_market customers"), ["mid_market", "customers"])


if __name__ == "__main__":
    unittest.main()
