"""Deterministic stand-in for an LLM that writes SQL.

Why it exists: the agent loop, the guard, the executor and the eval harness must be testable
without credentials or network, and the numbers in the README must be reproducible by anyone
who clones the repo.

How it works: regex patterns over the question, each building a SQL string. It is a parser,
not a model, and it was written against the template phrasings in `t2s/data/generate.py`, so
it handles those well by construction. That is the point and also the limit: a mock score is
evidence that the *harness* is correct, never evidence that text-to-SQL is solved.

It is deliberately imperfect in the ways a real model is:

  * paraphrases it has no pattern for fall through to a plausible wrong query rather than an
    error, which is exactly how a confident model fails and the hardest failure to detect
  * negation is under-supported, and one negation template silently drops the date predicate
  * questions naming no known entity return needs_clarification instead of a guess

The failure profile is concentrated in paraphrases and negation on purpose; the eval report
breaks it down that way and the README says so.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from ..models import TransientError
from .base import ERROR_MARKER, LINKED_MARKER, PREV_SQL_MARKER, Q_MARKER, LLMResponse, after_marker

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Words that imply a known entity. If a question contains none of these it is too vague to
# answer and the mock asks instead of guessing.
_KNOWN_WORDS = frozenset("""
ticket tickets job jobs case cases customer customers client clients account accounts
equipment machine machines unit units asset assets model models technician technicians
engineer engineers part parts spare spares component components fault faults priority
status city cities region regions segment cost costs price prices quantity install
installed opened closed open close raised resolved used usage consumed enterprise
mid_market small west north south east low medium high critical
""".split())


def _month_range(month: str, year: str) -> tuple[str, str]:
    """[start, end) for a named month, as ISO dates. Half-open so no day is double counted."""
    m = _MONTHS[month.lower()]
    y = int(year)
    end_y, end_m = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}-{m:02d}-01", f"{end_y:04d}-{end_m:02d}-01"


class MockLLM:
    def __init__(self, fail_first_n_calls: int = 0):
        self._fail_remaining = fail_first_n_calls
        self.calls = 0

    # ---- public contract ---------------------------------------------------
    def complete(self, task: str, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        self.calls += 1
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise TransientError("mock: simulated rate limit (429)")

        question = after_marker(user, Q_MARKER)
        if task == "generate":
            payload = self._generate(question, after_marker(user, LINKED_MARKER))
        elif task == "correct":
            payload = self._correct(
                question,
                after_marker(user, PREV_SQL_MARKER),
                after_marker(user, ERROR_MARKER),
                after_marker(user, LINKED_MARKER),
            )
        else:
            raise ValueError(f"mock has no behaviour for task={task!r}")

        content = json.dumps(payload)
        return LLMResponse(
            content=content,
            prompt_tokens=_approx_tokens(system) + _approx_tokens(user),
            completion_tokens=_approx_tokens(content),
            model="mock",
        )

    # ---- generation --------------------------------------------------------
    def _generate(self, question: str, linked: str) -> dict:
        q = " ".join(question.lower().replace("?", " ").split())

        for pattern, build in _RULES:
            m = pattern.search(q)
            if m is not None:
                sql = build(m)
                if sql:
                    return {"sql": sql, "confidence": 0.92,
                            "needs_clarification": False, "clarifying_question": None}

        # No pattern matched. Two very different situations, and conflating them is the bug
        # most text-to-SQL demos ship with.
        if not (set(re.findall(r"[a-z_]+", q)) & _KNOWN_WORDS):
            return {
                "sql": None, "confidence": 0.0, "needs_clarification": True,
                "clarifying_question": (
                    "I could not tell which tickets, customers, equipment or parts you mean. "
                    "Which of those are you asking about, and over what time period?"
                ),
            }

        # The question names something real but is phrased in a way the parser does not cover.
        # A real model does not stop here: it returns a confident, plausible, wrong query. So
        # does this, because a harness that never sees that failure cannot measure it.
        table = _first_linked_table(linked) or "service_tickets"
        return {"sql": f"SELECT * FROM {table}", "confidence": 0.34,
                "needs_clarification": False, "clarifying_question": None}

    # ---- one correction round ----------------------------------------------
    def _correct(self, question: str, prev_sql: str, error: str, linked: str) -> dict:
        """Fix the errors a schema-grounded model actually makes: a column on the wrong table,
        or a name that does not exist. Anything else, it admits it cannot fix."""
        err = error.lower()
        sql = prev_sql

        m = re.search(r"no such column:\s*([a-z_0-9.]+)", err)
        if m:
            bad = m.group(1).split(".")[-1]
            fixed = _COLUMN_FIXES.get(bad)
            if fixed:
                sql = re.sub(rf"\b{re.escape(bad)}\b", fixed, sql)
                return {"sql": sql, "confidence": 0.7,
                        "needs_clarification": False, "clarifying_question": None}

        m = re.search(r"no such table:\s*([a-z_0-9]+)", err)
        if m:
            bad = m.group(1)
            fixed = _TABLE_FIXES.get(bad)
            if fixed:
                sql = re.sub(rf"\b{re.escape(bad)}\b", fixed, sql)
                return {"sql": sql, "confidence": 0.7,
                        "needs_clarification": False, "clarifying_question": None}

        return {"sql": None, "confidence": 0.0, "needs_clarification": False,
                "clarifying_question": None}


# Singular/plural and near-miss names a model reaches for. Used only by the correction round.
_COLUMN_FIXES = {
    "customer_name": "name", "technician_name": "name", "part_name": "name",
    "open_date": "opened_at", "close_date": "closed_at", "opened": "opened_at",
    "closed": "closed_at", "cost": "unit_cost", "qty": "quantity",
}
_TABLE_FIXES = {
    "tickets": "service_tickets", "ticket": "service_tickets", "customer": "customers",
    "technician": "technicians", "part": "parts", "machine": "equipment",
}


def _first_linked_table(linked: str) -> Optional[str]:
    names = [n.strip() for n in linked.split(",") if n.strip()]
    return names[0] if names else None


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


# ---- the rule table ----------------------------------------------------------------
# Order matters: more specific patterns first. Each entry is (regex, builder).
_RULES: list[tuple[re.Pattern, Callable[[re.Match], Optional[str]]]] = []


def _rule(pattern: str) -> Callable:
    def register(fn: Callable[[re.Match], Optional[str]]):
        _RULES.append((re.compile(pattern), fn))
        return fn
    return register


# --- group by + top-k -----------------------------------------------------------------
@_rule(r"top (\d+) fault codes by ticket count|the (\d+) most common fault codes")
def _top_fault_codes(m):
    k = m.group(1) or m.group(2)
    return ("SELECT fault_code, COUNT(*) AS n FROM service_tickets "
            f"GROUP BY fault_code ORDER BY n DESC, fault_code ASC LIMIT {k}")


@_rule(r"top (\d+) technicians by closed tickets|which (\d+) technicians closed the most tickets")
def _top_technicians(m):
    k = m.group(1) or m.group(2)
    return ("SELECT t.technician_id, t.name, COUNT(*) AS n "
            "FROM technicians t JOIN service_tickets st ON t.technician_id = st.technician_id "
            "WHERE st.status = 'closed' GROUP BY t.technician_id, t.name "
            f"ORDER BY n DESC, t.technician_id ASC LIMIT {k}")


@_rule(r"top (\d+) customers by ticket count|the (\d+) customers with the most tickets")
def _top_customers(m):
    k = m.group(1) or m.group(2)
    return ("SELECT c.customer_id, c.name, COUNT(*) AS n FROM customers c "
            "JOIN equipment e ON c.customer_id = e.customer_id "
            "JOIN service_tickets st ON e.equipment_id = st.equipment_id "
            "GROUP BY c.customer_id, c.name ORDER BY n DESC, c.customer_id ASC "
            f"LIMIT {k}")


@_rule(r"top (\d+) parts by total quantity used|the (\d+) most used parts")
def _top_parts(m):
    k = m.group(1) or m.group(2)
    return ("SELECT p.part_id, p.name, SUM(tp.quantity) AS total_quantity FROM parts p "
            "JOIN ticket_parts tp ON p.part_id = tp.part_id GROUP BY p.part_id, p.name "
            f"ORDER BY total_quantity DESC, p.part_id ASC LIMIT {k}")


@_rule(r"top (\d+) cities by ticket count|which (\d+) cities raised the most tickets")
def _top_cities(m):
    k = m.group(1) or m.group(2)
    return ("SELECT c.city, COUNT(*) AS n FROM customers c "
            "JOIN equipment e ON c.customer_id = e.customer_id "
            "JOIN service_tickets st ON e.equipment_id = st.equipment_id "
            f"GROUP BY c.city ORDER BY n DESC, c.city ASC LIMIT {k}")


@_rule(r"top (\d+) equipment models by ticket count|the (\d+) models with the most tickets")
def _top_models(m):
    k = m.group(1) or m.group(2)
    return ("SELECT e.model, COUNT(*) AS n FROM equipment e "
            "JOIN service_tickets st ON e.equipment_id = st.equipment_id "
            f"GROUP BY e.model ORDER BY n DESC, e.model ASC LIMIT {k}")


# --- aggregate ------------------------------------------------------------------------
@_rule(r"average time to close a ticket by priority|mean days to close tickets for each priority")
def _avg_close_by_priority(_m):
    return ("SELECT priority, ROUND(AVG(julianday(closed_at) - julianday(opened_at)), 2) AS avg_days "
            "FROM service_tickets WHERE closed_at IS NOT NULL GROUP BY priority ORDER BY priority ASC")


@_rule(r"average time to close a ticket by region|mean days to close per technician region")
def _avg_close_by_region(_m):
    return ("SELECT t.region, ROUND(AVG(julianday(st.closed_at) - julianday(st.opened_at)), 2) AS avg_days "
            "FROM service_tickets st JOIN technicians t ON st.technician_id = t.technician_id "
            "WHERE st.closed_at IS NOT NULL GROUP BY t.region ORDER BY t.region ASC")


@_rule(r"average unit cost of parts|mean part price")
def _avg_part_cost(_m):
    return "SELECT ROUND(AVG(unit_cost), 2) AS avg_unit_cost FROM parts"


@_rule(r"total quantity of parts used|sum of part quantities used")
def _total_quantity(_m):
    return "SELECT SUM(quantity) AS total_quantity FROM ticket_parts"


@_rule(r"the most expensive part|which part has the highest unit cost")
def _most_expensive_part(_m):
    return "SELECT part_id, name, unit_cost FROM parts ORDER BY unit_cost DESC, part_id ASC LIMIT 1"


@_rule(r"total parts cost per ticket|parts spend for each ticket")
def _parts_cost_per_ticket(_m):
    return ("SELECT tp.ticket_id, ROUND(SUM(tp.quantity * p.unit_cost), 2) AS parts_cost "
            "FROM ticket_parts tp JOIN parts p ON tp.part_id = p.part_id "
            "GROUP BY tp.ticket_id ORDER BY tp.ticket_id ASC")


# --- join -----------------------------------------------------------------------------
@_rule(r"technicians who closed tickets for customers in ([a-z ]+?)\s*$")
def _techs_closed_for_city(m):
    return ("SELECT DISTINCT t.technician_id, t.name FROM technicians t "
            "JOIN service_tickets st ON t.technician_id = st.technician_id "
            "JOIN equipment e ON st.equipment_id = e.equipment_id "
            "JOIN customers c ON e.customer_id = c.customer_id "
            f"WHERE c.city = '{_title(m.group(1))}' AND st.status = 'closed'")


@_rule(r"equipment belonging to customers in the ([a-z_]+) segment")
def _equipment_by_segment(m):
    return ("SELECT e.equipment_id, e.model FROM equipment e "
            "JOIN customers c ON e.customer_id = c.customer_id "
            f"WHERE c.segment = '{m.group(1)}'")


@_rule(r"tickets for equipment model (.+?)\s*$")
def _tickets_for_model(m):
    return ("SELECT st.ticket_id, e.model FROM service_tickets st "
            "JOIN equipment e ON st.equipment_id = e.equipment_id "
            f"WHERE e.model = '{_model_case(m.group(1))}'")


@_rule(r"parts used on tickets with priority ([a-z]+)")
def _parts_by_priority(m):
    return ("SELECT DISTINCT p.part_id, p.name FROM parts p "
            "JOIN ticket_parts tp ON p.part_id = tp.part_id "
            "JOIN service_tickets st ON tp.ticket_id = st.ticket_id "
            f"WHERE st.priority = '{m.group(1)}'")


@_rule(r"customers whose equipment had an? ([a-z]+) priority ticket")
def _customers_with_priority(m):
    return ("SELECT DISTINCT c.customer_id, c.name FROM customers c "
            "JOIN equipment e ON c.customer_id = e.customer_id "
            "JOIN service_tickets st ON e.equipment_id = st.equipment_id "
            f"WHERE st.priority = '{m.group(1)}'")


@_rule(r"technicians who worked on equipment model (.+?)\s*$")
def _techs_for_model(m):
    return ("SELECT DISTINCT t.technician_id, t.name FROM technicians t "
            "JOIN service_tickets st ON t.technician_id = st.technician_id "
            "JOIN equipment e ON st.equipment_id = e.equipment_id "
            f"WHERE e.model = '{_model_case(m.group(1))}'")


# --- date range -----------------------------------------------------------------------
@_rule(r"tickets opened between (\d{4}-\d{2}-\d{2}) and (\d{4}-\d{2}-\d{2})")
def _tickets_between(m):
    return ("SELECT ticket_id, opened_at FROM service_tickets "
            f"WHERE opened_at >= '{m.group(1)}' AND opened_at <= '{m.group(2)}'")


@_rule(r"parts used on tickets closed in the last quarter of (\d{4})")
def _parts_last_quarter(m):
    y = m.group(1)
    return ("SELECT DISTINCT p.part_id, p.name FROM parts p "
            "JOIN ticket_parts tp ON p.part_id = tp.part_id "
            "JOIN service_tickets st ON tp.ticket_id = st.ticket_id "
            f"WHERE st.closed_at >= '{y}-10-01' AND st.closed_at < '{int(y)+1}-01-01'")


@_rule(r"tickets closed in (\d{4})\s*$")
def _tickets_closed_in_year(m):
    y = m.group(1)
    return ("SELECT ticket_id, closed_at FROM service_tickets "
            f"WHERE closed_at >= '{y}-01-01' AND closed_at < '{int(y)+1}-01-01'")


@_rule(r"equipment installed before (\d{4}-\d{2}-\d{2})")
def _equipment_before(m):
    return f"SELECT equipment_id, model, install_date FROM equipment WHERE install_date < '{m.group(1)}'"


@_rule(r"tickets opened in the first half of (\d{4})")
def _tickets_first_half(m):
    y = m.group(1)
    return ("SELECT ticket_id, opened_at FROM service_tickets "
            f"WHERE opened_at >= '{y}-01-01' AND opened_at < '{y}-07-01'")


# --- count ----------------------------------------------------------------------------
@_rule(r"how many tickets were opened in ([a-z]+) (\d{4})|count of tickets opened during ([a-z]+) (\d{4})")
def _count_tickets_month(m):
    month = m.group(1) or m.group(3)
    year = m.group(2) or m.group(4)
    if month.lower() not in _MONTHS:
        return None
    lo, hi = _month_range(month, year)
    return f"SELECT COUNT(*) AS n FROM service_tickets WHERE opened_at >= '{lo}' AND opened_at < '{hi}'"


@_rule(r"how many tickets are still open|number of open tickets")
def _count_open(_m):
    return "SELECT COUNT(*) AS n FROM service_tickets WHERE status = 'open'"


@_rule(r"how many customers are in ([a-z ]+?)\s*$|count the customers based in ([a-z ]+?)\s*$")
def _count_customers_city(m):
    city = m.group(1) or m.group(2)
    return f"SELECT COUNT(*) AS n FROM customers WHERE city = '{_title(city)}'"


@_rule(r"how many technicians work in the ([a-z]+) region|number of technicians in ([a-z]+)\s*$")
def _count_technicians_region(m):
    region = m.group(1) or m.group(2)
    return f"SELECT COUNT(*) AS n FROM technicians WHERE region = '{_title(region)}'"


@_rule(r"how many tickets have priority ([a-z]+)|count of ([a-z]+) priority tickets")
def _count_by_priority(m):
    return f"SELECT COUNT(*) AS n FROM service_tickets WHERE priority = '{m.group(1) or m.group(2)}'"


@_rule(r"how many distinct fault codes were reported|number of different fault codes seen")
def _count_fault_codes(_m):
    return "SELECT COUNT(DISTINCT fault_code) AS n FROM service_tickets"


# --- negation -------------------------------------------------------------------------
# Under-supported on purpose. Three of the five templates are handled correctly, one is
# handled *wrongly* in the way models actually fail here, and one is not handled at all.
@_rule(r"customers with no tickets in (\d{4})|which customers did not raise a ticket in (\d{4})")
def _customers_no_tickets_in_year(_m):
    # The bug: the year is parsed out of the question and then never used. The query answers
    # "customers with no tickets ever", which is a strict subset, so it returns too few rows
    # and looks entirely reasonable while doing it. This is the single most common
    # text-to-SQL failure that unit tests do not catch and execution accuracy does.
    return ("SELECT c.customer_id, c.name FROM customers c WHERE NOT EXISTS ("
            "SELECT 1 FROM equipment e JOIN service_tickets st ON e.equipment_id = st.equipment_id "
            "WHERE e.customer_id = c.customer_id)")


@_rule(r"technicians who have not closed any tickets|which technicians never closed a ticket")
def _technicians_no_closed(_m):
    return ("SELECT t.technician_id, t.name FROM technicians t WHERE NOT EXISTS ("
            "SELECT 1 FROM service_tickets st WHERE st.technician_id = t.technician_id "
            "AND st.status = 'closed')")


@_rule(r"equipment with no service tickets|machines that never needed service")
def _equipment_no_tickets(_m):
    return ("SELECT e.equipment_id, e.model FROM equipment e WHERE NOT EXISTS ("
            "SELECT 1 FROM service_tickets st WHERE st.equipment_id = e.equipment_id)")


@_rule(r"parts never used on any ticket|which parts have never been consumed")
def _parts_never_used(_m):
    return ("SELECT p.part_id, p.name FROM parts p WHERE NOT EXISTS ("
            "SELECT 1 FROM ticket_parts tp WHERE tp.part_id = p.part_id)")


# --- simple filter --------------------------------------------------------------------
@_rule(r"tickets with priority ([a-z]+) that are still open|which ([a-z]+) priority tickets are still open")
def _tickets_priority_open(m):
    p = m.group(1) or m.group(2)
    return ("SELECT ticket_id, priority, status FROM service_tickets "
            f"WHERE priority = '{p}' AND status = 'open'")


@_rule(r"technicians in the ([a-z]+) region|list technicians working in ([a-z]+)\s*$")
def _technicians_region(m):
    region = m.group(1) or m.group(2)
    return f"SELECT technician_id, name, region FROM technicians WHERE region = '{_title(region)}'"


@_rule(r"customers in the ([a-z_]+) segment|which customers are ([a-z_]+) accounts")
def _customers_segment(m):
    return f"SELECT customer_id, name, segment FROM customers WHERE segment = '{m.group(1) or m.group(2)}'"


@_rule(r"parts that cost more than (\d+)|parts with a unit cost above (\d+)")
def _parts_cost_above(m):
    return f"SELECT part_id, name, unit_cost FROM parts WHERE unit_cost > {m.group(1) or m.group(2)}"


@_rule(r"tickets with fault code (f-\d+)|show tickets reporting fault (f-\d+)")
def _tickets_fault_code(m):
    return ("SELECT ticket_id, fault_code, status FROM service_tickets "
            f"WHERE fault_code = '{(m.group(1) or m.group(2)).upper()}'")


# Last among the filters: "customers in X" would otherwise swallow the segment form above.
@_rule(r"^customers in ([a-z ]+?)\s*$|which customers are based in ([a-z ]+?)\s*$")
def _customers_city(m):
    return f"SELECT customer_id, name, city FROM customers WHERE city = '{_title(m.group(1) or m.group(2))}'"


def _title(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())


def _model_case(s: str) -> str:
    """Equipment models render as 'Helios 400': capitalised word plus digits."""
    return " ".join(w.upper() if w.isdigit() else w.capitalize() for w in s.split())
