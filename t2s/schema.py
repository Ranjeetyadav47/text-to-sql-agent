"""Schema loading, the foreign-key graph, and schema linking.

Schema linking is the lever this repo is really about. A six-table schema fits in a prompt,
so linking looks pointless here; at forty tables it is the difference between a prompt that
fits and one that does not, and between a model that picks the right join and one that picks
a plausible wrong one. The eval measures the effect directly (`eval --compare-linking`)
rather than asserting it.

The structure is read from SQLite itself (PRAGMA), so it cannot drift from the database.
The column descriptions are curated here: they are documentation written for the model, and
SQLite has nowhere to put them.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Iterable, Optional

from .config import LinkConfig
from .models import Column, LinkedSchema, Table

# What each table and column means, in the words a user would use. These are part of the
# prompt and part of the linking signal, so vocabulary here is a design decision: every
# word a user is likely to say should appear in some description.
TABLE_DOCS: dict[str, str] = {
    "customers": "Companies that own serviced equipment. One row per customer account.",
    "equipment": "Individual machines installed at a customer site. One row per unit.",
    "technicians": "Field service engineers who attend tickets. One row per technician.",
    "service_tickets": "Service jobs raised against a machine. One row per ticket, opened and later closed.",
    "parts": "Spare parts catalogue with unit cost. One row per part.",
    "ticket_parts": "Which parts were consumed on which ticket, and how many.",
}

COLUMN_DOCS: dict[str, str] = {
    "customers.customer_id": "Unique customer identifier.",
    "customers.name": "Customer company name.",
    "customers.city": "City the customer is based in, for example Pune or Nashik.",
    "customers.segment": "Commercial segment: enterprise, mid_market or small.",
    "equipment.equipment_id": "Unique machine identifier.",
    "equipment.customer_id": "Customer that owns this machine.",
    "equipment.model": "Equipment model name, for example Helios 400.",
    "equipment.install_date": "Date the machine was installed and commissioned.",
    "technicians.technician_id": "Unique technician identifier.",
    "technicians.name": "Technician full name.",
    "technicians.region": "Sales and service region the technician covers: West, North, South or East.",
    "service_tickets.ticket_id": "Unique ticket identifier.",
    "service_tickets.equipment_id": "Machine the ticket was raised against.",
    "service_tickets.technician_id": "Technician assigned to the ticket.",
    "service_tickets.opened_at": "Date the ticket was opened or raised.",
    "service_tickets.closed_at": "Date the ticket was closed or resolved, null while still open.",
    "service_tickets.priority": "Ticket urgency: low, medium, high or critical.",
    "service_tickets.status": "Ticket state: open, in_progress, closed or cancelled.",
    "service_tickets.fault_code": "Diagnostic fault code reported on the machine, for example F-101.",
    "parts.part_id": "Unique part identifier.",
    "parts.name": "Part name as it appears in the spares catalogue.",
    "parts.unit_cost": "Cost of one unit of the part, in INR.",
    "ticket_parts.ticket_id": "Ticket the part was used on.",
    "ticket_parts.part_id": "Part that was used.",
    "ticket_parts.quantity": "How many units of the part were consumed.",
}

# Words users say that are not column names. Without these, "machines that never needed
# service" links nothing, because the schema never uses the word "machine".
SYNONYMS: dict[str, tuple[str, ...]] = {
    "customers": ("customer", "customers", "client", "clients", "account", "accounts", "company", "companies",
                  "city", "cities", "segment", "enterprise", "mid_market", "small", "based"),
    "equipment": ("equipment", "machine", "machines", "unit", "units", "asset", "assets", "model", "models",
                  "installed", "install", "commissioned"),
    "technicians": ("technician", "technicians", "engineer", "engineers", "region", "regions", "west", "north",
                    "south", "east", "staff"),
    "service_tickets": ("ticket", "tickets", "job", "jobs", "case", "cases", "fault", "faults", "priority",
                        "priorities", "status", "open", "opened", "closed", "close", "raised", "resolved",
                        "critical", "high", "medium", "low", "service"),
    "parts": ("part", "parts", "spare", "spares", "component", "components", "cost", "costs", "price",
              "prices", "expensive", "catalogue"),
    "ticket_parts": ("used", "usage", "consumed", "quantity", "quantities", "consumption", "fitted"),
}

_STOPWORDS = frozenset("""
a an the and or of for in on at to from by with is are was were be been being do does did how
what which who whom whose when where why me my we our you your show list give tell find get
many much any all some each per most least more less than that this these those it its as
""".split())

_WORD = re.compile(r"[a-z_][a-z0-9_]*")


def tokenise(question: str) -> list[str]:
    """Lowercase word tokens with stopwords removed. Keeps underscores so `mid_market` survives."""
    return [w for w in _WORD.findall(question.lower()) if w not in _STOPWORDS]


class Schema:
    """The full database schema, read from SQLite."""

    def __init__(self, tables: list[Table], foreign_keys: list[tuple[str, str, str, str]]):
        self.tables = tables
        self.by_name = {t.name: t for t in tables}
        # (from_table, from_col, to_table, to_col)
        self.foreign_keys = foreign_keys
        self._neighbours: dict[str, set[str]] = {t.name: set() for t in tables}
        for ft, _fc, tt, _tc in foreign_keys:
            if ft in self._neighbours and tt in self._neighbours:
                self._neighbours[ft].add(tt)
                self._neighbours[tt].add(ft)

    @classmethod
    def from_sqlite(cls, db_path: str) -> "Schema":
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            names = [r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
            tables: list[Table] = []
            fks: list[tuple[str, str, str, str]] = []
            for name in names:
                cols: list[Column] = []
                for r in conn.execute(f"PRAGMA table_info({name})"):
                    cols.append(Column(
                        name=r["name"],
                        type=r["type"] or "",
                        description=COLUMN_DOCS.get(f"{name}.{r['name']}", ""),
                        is_pk=bool(r["pk"]),
                    ))
                for r in conn.execute(f"PRAGMA foreign_key_list({name})"):
                    fks.append((name, r["from"], r["table"], r["to"]))
                    col = next((c for c in cols if c.name == r["from"]), None)
                    if col:
                        col.references = (r["table"], r["to"])
                tables.append(Table(name=name, description=TABLE_DOCS.get(name, ""), columns=cols))
            return cls(tables, fks)
        finally:
            conn.close()

    def neighbours(self, table: str) -> set[str]:
        return self._neighbours.get(table, set())

    def join_hint(self, a: str, b: str) -> Optional[str]:
        """The FK that joins two tables, as a readable equality, or None if they do not join."""
        for ft, fc, tt, tc in self.foreign_keys:
            if (ft, tt) == (a, b):
                return f"{ft}.{fc} = {tt}.{tc}"
            if (ft, tt) == (b, a):
                return f"{ft}.{fc} = {tt}.{tc}"
        return None

    # ---- rendering -------------------------------------------------------------
    def render(self, tables: Optional[Iterable[str]] = None) -> str:
        """DDL-ish text for the prompt. Compact on purpose: this is tokens on every call."""
        wanted = list(tables) if tables is not None else [t.name for t in self.tables]
        out: list[str] = []
        for name in wanted:
            t = self.by_name.get(name)
            if t is None:
                continue
            out.append(f"TABLE {t.name}  -- {t.description}")
            for c in t.columns:
                bits = [f"  {c.name} {c.type}"]
                if c.is_pk:
                    bits.append("PRIMARY KEY")
                if c.references:
                    bits.append(f"REFERENCES {c.references[0]}({c.references[1]})")
                line = " ".join(bits)
                if c.description:
                    line = f"{line:<52}-- {c.description}"
                out.append(line)
        rels = [f"  {a} -> {b}: {self.join_hint(a, b)}" for a in wanted for b in wanted
                if a < b and self.join_hint(a, b)]
        if rels:
            out.append("RELATIONSHIPS")
            out.extend(rels)
        return "\n".join(out)


class SchemaLinker:
    """Pick the few tables a question is about.

    Deliberately not an LLM call: it is keyword overlap against column names and the curated
    descriptions, plus one hop along the foreign-key graph. It is cheap, deterministic, and
    its mistakes are legible, which matters more than its ceiling.
    """

    def __init__(self, schema: Schema, cfg: LinkConfig | None = None):
        self.schema = schema
        self.cfg = cfg or LinkConfig()
        self._vocab = {t.name: self._table_vocab(t) for t in schema.tables}

    def _table_vocab(self, table: Table) -> set[str]:
        words: set[str] = set(tokenise(table.name.replace("_", " ")))
        words.update(tokenise(table.description))
        for c in table.columns:
            words.update(tokenise(c.name.replace("_", " ")))
            words.update(tokenise(c.description))
        words.update(SYNONYMS.get(table.name, ()))
        return words - _STOPWORDS

    def score(self, question: str) -> dict[str, int]:
        tokens = set(tokenise(question))
        return {name: len(tokens & vocab) for name, vocab in self._vocab.items()}

    def link(self, question: str) -> LinkedSchema:
        scores = self.score(question)
        considered = len(self.schema.tables)
        seeds = [n for n, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                 if s >= self.cfg.min_keyword_hits]

        if not seeds:
            # Nothing matched. Returning an empty link is the honest answer: the agent turns
            # it into a clarifying question rather than guessing at the whole schema.
            return LinkedSchema(tables=[], reasons={}, considered=considered)

        chosen: list[str] = []
        reasons: dict[str, str] = {}
        for name in seeds:
            if len(chosen) >= self.cfg.max_tables:
                break
            chosen.append(name)
            reasons[name] = f"matched {scores[name]} schema keyword(s)"

        # One hop along the FK graph: a question about tickets and cities needs `equipment`
        # in between, and no keyword in the question will ever name it.
        if self.cfg.follow_foreign_keys:
            for name in list(chosen):
                for nb in sorted(self.schema.neighbours(name)):
                    if len(chosen) >= self.cfg.max_tables:
                        break
                    if nb not in chosen and self._bridges(nb, chosen):
                        chosen.append(nb)
                        reasons[nb] = f"foreign-key bridge between {name} and the rest of the query"

        return LinkedSchema(
            tables=[self.schema.by_name[n] for n in chosen],
            reasons=reasons,
            considered=considered,
        )

    def _bridges(self, candidate: str, chosen: list[str]) -> bool:
        """True if `candidate` connects two already-chosen tables that do not join directly."""
        linked = [c for c in chosen if self.schema.join_hint(candidate, c)]
        if len(linked) < 2:
            return False
        return any(self.schema.join_hint(a, b) is None for a in linked for b in linked if a < b)
