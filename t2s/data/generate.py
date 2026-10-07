"""Seeded synthetic field-service database and the labelled question set.

Everything here is invented. No schema, name, model, part or number comes from any employer
or client, and the whole database is rebuilt from this file by `python -m t2s generate-data`.

Two outputs:

  * `service.db`      - the database the agent queries
  * `questions.jsonl` - 90 questions with gold SQL and a category label

Gold *results* are never stored, only gold SQL. The eval runs the gold query at scoring time
and compares result sets, so the labels cannot drift away from the database: regenerate the
database and the labels regenerate with it. Storing expected rows would have been easier and
would have rotted the first time the seed changed.

Each template question is also emitted as a paraphrase. The paraphrases are hand-written
here rather than produced by a model, because a model-written paraphrase set tests whether
two models agree, not whether the agent understands a rephrasing.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import date, timedelta
from typing import Iterator

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    city         TEXT    NOT NULL,
    segment      TEXT    NOT NULL
);

CREATE TABLE equipment (
    equipment_id INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    model        TEXT    NOT NULL,
    install_date TEXT    NOT NULL
);

CREATE TABLE technicians (
    technician_id INTEGER PRIMARY KEY,
    name          TEXT   NOT NULL,
    region        TEXT   NOT NULL
);

CREATE TABLE service_tickets (
    ticket_id     INTEGER PRIMARY KEY,
    equipment_id  INTEGER NOT NULL REFERENCES equipment(equipment_id),
    technician_id INTEGER NOT NULL REFERENCES technicians(technician_id),
    opened_at     TEXT    NOT NULL,
    closed_at     TEXT,
    priority      TEXT    NOT NULL,
    status        TEXT    NOT NULL,
    fault_code    TEXT    NOT NULL
);

CREATE TABLE parts (
    part_id   INTEGER PRIMARY KEY,
    name      TEXT    NOT NULL,
    unit_cost REAL    NOT NULL
);

CREATE TABLE ticket_parts (
    ticket_id INTEGER NOT NULL REFERENCES service_tickets(ticket_id),
    part_id   INTEGER NOT NULL REFERENCES parts(part_id),
    quantity  INTEGER NOT NULL,
    PRIMARY KEY (ticket_id, part_id)
);

CREATE INDEX idx_equipment_customer ON equipment(customer_id);
CREATE INDEX idx_tickets_equipment  ON service_tickets(equipment_id);
CREATE INDEX idx_tickets_technician ON service_tickets(technician_id);
CREATE INDEX idx_tickets_opened     ON service_tickets(opened_at);
"""

CITIES = ["Pune", "Nashik", "Nagpur", "Mumbai", "Indore", "Surat", "Vadodara", "Jaipur"]
SEGMENTS = ["enterprise", "mid_market", "small"]
REGIONS = ["West", "North", "South", "East"]
PRIORITIES = ["low", "medium", "high", "critical"]
STATUSES = ["open", "in_progress", "closed", "cancelled"]
MODELS = ["Helios 400", "Terra 55", "Vertex 120", "Orbit 9",
          "Cascade 300", "Pinnacle 75", "Quartz 210", "Zenith 60"]
FAULT_CODES = [f"F-{100 + i}" for i in range(1, 19)]

_CO_HEAD = ["Anchor", "Bluepeak", "Cedarline", "Dovetail", "Eastgate", "Fernwood", "Granite",
            "Harbourstone", "Ironleaf", "Juniper", "Kingfisher", "Larkspur", "Milestone",
            "Northbay", "Oakfield", "Pinecrest", "Quarrystone", "Riverbend", "Stonebridge",
            "Thornfield", "Umberwood", "Valecrest", "Westmoor", "Yellowstone", "Zephyr"]
_CO_TAIL = ["Agro Works", "Industries", "Mills", "Processing", "Foods", "Textiles",
            "Chemicals", "Packaging", "Dairy", "Beverages"]

_FIRST = ["Aarav", "Bhavna", "Chetan", "Divya", "Eshan", "Farida", "Gaurav", "Hema", "Imran",
          "Jyoti", "Kabir", "Lata", "Manish", "Neha", "Omkar", "Pooja", "Rahul", "Sneha",
          "Tarun", "Usha", "Vikram", "Waseem", "Yamini", "Zoya", "Nikhil"]
_LAST = ["Apte", "Bhat", "Chawla", "Deshmukh", "Engineer", "Fernandes", "Gokhale", "Hegde",
         "Iyer", "Joshi", "Kulkarni", "Lamba", "Mehta", "Nair", "Oberoi", "Pandit", "Rao",
         "Sethi", "Tandon", "Upadhyay", "Verma", "Wagh", "Yadav", "Zaveri", "Bose"]

_PART_HEAD = ["Drive", "Pressure", "Hydraulic", "Rotor", "Intake", "Exhaust", "Coolant",
              "Bearing", "Spindle", "Gasket", "Control", "Sensor", "Filter", "Valve",
              "Coupling", "Actuator", "Pulley", "Thermal", "Vibration", "Feed"]
_PART_TAIL = ["belt", "seal", "assembly", "module", "cartridge", "housing", "kit", "plate"]

# Fixed parameters the question set asks about. Pinned rather than sampled so the questions
# are stable across regeneration, and chosen to be values the generator always produces.
Q_CITY = "Pune"
Q_REGION = "West"
Q_SEGMENT = "enterprise"
Q_PRIORITY = "high"
Q_FAULT = "F-101"
Q_MODEL = "Helios 400"
Q_COST = 5000
Q_MONTH, Q_MONTH_YEAR = "March", "2026"
Q_DATE_FROM, Q_DATE_TO = "2025-03-01", "2025-06-30"
Q_INSTALL_BEFORE = "2025-06-01"

START = date(2025, 1, 1)
END = date(2026, 12, 31)

# Reserved groups whose whole purpose is to give the negation questions gold answers that a
# broken query cannot match by accident. See the comment in `_populate`.
LEGACY_CUSTOMERS = (44, 45, 46, 47)
LEGACY_EQUIPMENT = range(180, 189)      # ticketed, but only during 2025
UNTICKETED_EQUIPMENT = range(189, 201)  # never ticketed at all


def generate(out_dir: str = "data", seed: int = 7) -> dict[str, int]:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "service.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        counts = _populate(conn, rng)
        conn.commit()
    finally:
        conn.close()

    questions = build_questions()
    with open(os.path.join(out_dir, "questions.jsonl"), "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    counts["questions"] = len(questions)
    return counts


def _populate(conn: sqlite3.Connection, rng: random.Random) -> dict[str, int]:
    # Customers: every city is used, and Pune is guaranteed a healthy share because the
    # question set asks about it by name.
    customers = []
    for i in range(1, 51):
        head = _CO_HEAD[(i - 1) % len(_CO_HEAD)]
        tail = _CO_TAIL[(i * 3) % len(_CO_TAIL)]
        city = Q_CITY if i % 4 == 1 else rng.choice(CITIES)
        customers.append((i, f"{head} {tail}", city, rng.choice(SEGMENTS)))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)

    technicians = []
    for i in range(1, 26):
        name = f"{_FIRST[(i - 1) % len(_FIRST)]} {_LAST[(i * 7) % len(_LAST)]}"
        technicians.append((i, name, REGIONS[(i - 1) % len(REGIONS)]))
    conn.executemany("INSERT INTO technicians VALUES (?,?,?)", technicians)

    # Three reserved groups, each the gold answer to a question that would otherwise be
    # answerable by accident. A query returning everything, or nothing, must not score.
    #
    #   customers 48-50      own no equipment      -> "customers without equipment"
    #   equipment 180-188    ticketed in 2025 only -> "customers with no tickets in 2026"
    #   equipment 189-200    never ticketed        -> "equipment with no service tickets"
    #
    # The middle group is the important one. Without it, "no tickets in 2026" and "no tickets
    # ever" return the same rows, and a query that parses the year and then forgets to filter
    # on it scores as correct. That is the single most common text-to-SQL failure, so the
    # labelled set has to be able to see it.
    equipment = []
    for i in range(1, 201):
        install = date(2020, 1, 1) + timedelta(days=rng.randint(0, 2400))
        customer_id = (LEGACY_CUSTOMERS[(i - LEGACY_EQUIPMENT.start) % len(LEGACY_CUSTOMERS)]
                       if i in LEGACY_EQUIPMENT else rng.randint(1, 43))
        equipment.append((i, customer_id, MODELS[(i - 1) % len(MODELS)], install.isoformat()))
    conn.executemany("INSERT INTO equipment VALUES (?,?,?,?)", equipment)

    parts = []
    for i in range(1, 41):
        name = f"{_PART_HEAD[(i - 1) % len(_PART_HEAD)]} {_PART_TAIL[(i * 3) % len(_PART_TAIL)]}"
        parts.append((i, name, float(rng.randrange(500, 25_000, 50))))
    conn.executemany("INSERT INTO parts VALUES (?,?,?)", parts)

    # Technicians 23-25 never close a ticket, so "technicians who have not closed any" has a
    # real answer. They still attend open ones, which is what a recent joiner looks like.
    span = (END - START).days
    tickets = []
    for i in range(1, 2001):
        equipment_id = rng.randint(1, LEGACY_EQUIPMENT.stop - 1)
        if equipment_id in LEGACY_EQUIPMENT:
            opened = date(2025, 1, 1) + timedelta(days=rng.randint(0, 364))
        else:
            opened = START + timedelta(days=rng.randint(0, span))
        status = rng.choices(STATUSES, weights=[12, 12, 70, 6])[0]
        technician_id = rng.randint(1, 25)
        if status == "closed" and technician_id > 22:
            technician_id = rng.randint(1, 22)
        closed = None
        if status == "closed":
            # Keep the close date inside the window so "closed in 2026" means what it says.
            closed_date = opened + timedelta(days=rng.randint(1, 45))
            closed = min(closed_date, END).isoformat()
        tickets.append((
            i, equipment_id, technician_id, opened.isoformat(), closed,
            rng.choice(PRIORITIES), status, rng.choice(FAULT_CODES),
        ))
    conn.executemany("INSERT INTO service_tickets VALUES (?,?,?,?,?,?,?,?)", tickets)

    # Parts are consumed on roughly half of tickets. Leaving some parts never used is
    # deliberate: the negation category asks about exactly that.
    used = []
    for ticket_id in range(1, 2001):
        if rng.random() < 0.5:
            continue
        for part_id in rng.sample(range(1, 36), rng.randint(1, 3)):
            used.append((ticket_id, part_id, rng.randint(1, 6)))
    conn.executemany("INSERT INTO ticket_parts VALUES (?,?,?)", used)

    return {"customers": len(customers), "technicians": len(technicians),
            "equipment": len(equipment), "parts": len(parts),
            "service_tickets": len(tickets), "ticket_parts": len(used)}


def load_questions(data_dir: str = "data") -> Iterator[dict]:
    with open(os.path.join(data_dir, "questions.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- the labelled question set ------------------------------------------------------
# Each entry: (category, template question, paraphrase, gold SQL).
# Gold SQL is the correct answer, independent of what any model or the mock produces.

def _templates() -> list[tuple[str, str, str, str]]:
    return [
        # ---- simple filter ----------------------------------------------------------
        ("simple_filter",
         f"tickets with priority {Q_PRIORITY} that are still open",
         f"which {Q_PRIORITY} priority tickets are still open",
         "SELECT ticket_id, priority, status FROM service_tickets "
         f"WHERE priority = '{Q_PRIORITY}' AND status = 'open'"),
        ("simple_filter",
         f"technicians in the {Q_REGION} region",
         f"list technicians working in {Q_REGION}",
         f"SELECT technician_id, name, region FROM technicians WHERE region = '{Q_REGION}'"),
        ("simple_filter",
         f"customers in the {Q_SEGMENT} segment",
         f"which customers are {Q_SEGMENT} accounts",
         f"SELECT customer_id, name, segment FROM customers WHERE segment = '{Q_SEGMENT}'"),
        ("simple_filter",
         f"parts that cost more than {Q_COST}",
         f"parts costing above {Q_COST}",
         f"SELECT part_id, name, unit_cost FROM parts WHERE unit_cost > {Q_COST}"),
        ("simple_filter",
         f"tickets with fault code {Q_FAULT}",
         f"show tickets reporting fault {Q_FAULT}",
         f"SELECT ticket_id, fault_code, status FROM service_tickets WHERE fault_code = '{Q_FAULT}'"),
        ("simple_filter",
         f"customers in {Q_CITY}",
         f"where are our {Q_CITY} customers",
         f"SELECT customer_id, name, city FROM customers WHERE city = '{Q_CITY}'"),

        # ---- count ------------------------------------------------------------------
        ("count",
         f"how many tickets were opened in {Q_MONTH} {Q_MONTH_YEAR}",
         f"count of tickets opened during {Q_MONTH} {Q_MONTH_YEAR}",
         "SELECT COUNT(*) AS n FROM service_tickets "
         f"WHERE opened_at >= '{Q_MONTH_YEAR}-03-01' AND opened_at < '{Q_MONTH_YEAR}-04-01'"),
        ("count",
         "how many tickets are still open",
         "number of open tickets",
         "SELECT COUNT(*) AS n FROM service_tickets WHERE status = 'open'"),
        ("count",
         f"how many customers are in {Q_CITY}",
         f"count the customers based in {Q_CITY}",
         f"SELECT COUNT(*) AS n FROM customers WHERE city = '{Q_CITY}'"),
        ("count",
         f"how many technicians work in the {Q_REGION} region",
         f"number of technicians in {Q_REGION}",
         f"SELECT COUNT(*) AS n FROM technicians WHERE region = '{Q_REGION}'"),
        ("count",
         f"how many tickets have priority {Q_PRIORITY}",
         f"how many tickets came in at {Q_PRIORITY} priority",
         f"SELECT COUNT(*) AS n FROM service_tickets WHERE priority = '{Q_PRIORITY}'"),
        ("count",
         "how many distinct fault codes were reported",
         "how many unique fault codes are there",
         "SELECT COUNT(DISTINCT fault_code) AS n FROM service_tickets"),

        # ---- aggregate --------------------------------------------------------------
        ("aggregate",
         "average time to close a ticket by priority",
         "mean days to close tickets for each priority",
         "SELECT priority, ROUND(AVG(julianday(closed_at) - julianday(opened_at)), 2) AS avg_days "
         "FROM service_tickets WHERE closed_at IS NOT NULL GROUP BY priority ORDER BY priority ASC"),
        ("aggregate",
         "average unit cost of parts",
         "what is the mean part price",
         "SELECT ROUND(AVG(unit_cost), 2) AS avg_unit_cost FROM parts"),
        ("aggregate",
         "total quantity of parts used",
         "sum of part quantities used",
         "SELECT SUM(quantity) AS total_quantity FROM ticket_parts"),
        ("aggregate",
         "average time to close a ticket by region",
         "mean days to close per technician region",
         "SELECT t.region, ROUND(AVG(julianday(st.closed_at) - julianday(st.opened_at)), 2) AS avg_days "
         "FROM service_tickets st JOIN technicians t ON st.technician_id = t.technician_id "
         "WHERE st.closed_at IS NOT NULL GROUP BY t.region ORDER BY t.region ASC"),
        ("aggregate",
         "the most expensive part",
         "which part is priciest",
         "SELECT part_id, name, unit_cost FROM parts ORDER BY unit_cost DESC, part_id ASC LIMIT 1"),
        ("aggregate",
         "total parts cost per ticket",
         "parts spend for each ticket",
         "SELECT tp.ticket_id, ROUND(SUM(tp.quantity * p.unit_cost), 2) AS parts_cost "
         "FROM ticket_parts tp JOIN parts p ON tp.part_id = p.part_id "
         "GROUP BY tp.ticket_id ORDER BY tp.ticket_id ASC"),

        # ---- join -------------------------------------------------------------------
        ("join",
         f"technicians who closed tickets for customers in {Q_CITY}",
         f"which technicians handled closed tickets for {Q_CITY} customers",
         "SELECT DISTINCT t.technician_id, t.name FROM technicians t "
         "JOIN service_tickets st ON t.technician_id = st.technician_id "
         "JOIN equipment e ON st.equipment_id = e.equipment_id "
         "JOIN customers c ON e.customer_id = c.customer_id "
         f"WHERE c.city = '{Q_CITY}' AND st.status = 'closed'"),
        ("join",
         f"equipment belonging to customers in the {Q_SEGMENT} segment",
         f"which machines do {Q_SEGMENT} customers own",
         "SELECT e.equipment_id, e.model FROM equipment e "
         "JOIN customers c ON e.customer_id = c.customer_id "
         f"WHERE c.segment = '{Q_SEGMENT}'"),
        ("join",
         f"tickets for equipment model {Q_MODEL}",
         f"show tickets raised against {Q_MODEL} units",
         "SELECT st.ticket_id, e.model FROM service_tickets st "
         "JOIN equipment e ON st.equipment_id = e.equipment_id "
         f"WHERE e.model = '{Q_MODEL}'"),
        ("join",
         f"parts used on tickets with priority {Q_PRIORITY}",
         f"which parts went into {Q_PRIORITY} priority tickets",
         "SELECT DISTINCT p.part_id, p.name FROM parts p "
         "JOIN ticket_parts tp ON p.part_id = tp.part_id "
         "JOIN service_tickets st ON tp.ticket_id = st.ticket_id "
         f"WHERE st.priority = '{Q_PRIORITY}'"),
        ("join",
         f"customers whose equipment had a {Q_PRIORITY} priority ticket",
         f"which customers own equipment with {Q_PRIORITY} tickets",
         "SELECT DISTINCT c.customer_id, c.name FROM customers c "
         "JOIN equipment e ON c.customer_id = e.customer_id "
         "JOIN service_tickets st ON e.equipment_id = st.equipment_id "
         f"WHERE st.priority = '{Q_PRIORITY}'"),
        ("join",
         f"technicians who worked on equipment model {Q_MODEL}",
         f"which technicians serviced {Q_MODEL} machines",
         "SELECT DISTINCT t.technician_id, t.name FROM technicians t "
         "JOIN service_tickets st ON t.technician_id = st.technician_id "
         "JOIN equipment e ON st.equipment_id = e.equipment_id "
         f"WHERE e.model = '{Q_MODEL}'"),

        # ---- group by + top-k -------------------------------------------------------
        ("group_topk",
         "top 5 fault codes by ticket count",
         "the 5 most common fault codes",
         "SELECT fault_code, COUNT(*) AS n FROM service_tickets "
         "GROUP BY fault_code ORDER BY n DESC, fault_code ASC LIMIT 5"),
        ("group_topk",
         "top 3 technicians by closed tickets",
         "which 3 technicians closed the most tickets",
         "SELECT t.technician_id, t.name, COUNT(*) AS n "
         "FROM technicians t JOIN service_tickets st ON t.technician_id = st.technician_id "
         "WHERE st.status = 'closed' GROUP BY t.technician_id, t.name "
         "ORDER BY n DESC, t.technician_id ASC LIMIT 3"),
        ("group_topk",
         "top 5 customers by ticket count",
         "the 5 customers with the most tickets",
         "SELECT c.customer_id, c.name, COUNT(*) AS n FROM customers c "
         "JOIN equipment e ON c.customer_id = e.customer_id "
         "JOIN service_tickets st ON e.equipment_id = st.equipment_id "
         "GROUP BY c.customer_id, c.name ORDER BY n DESC, c.customer_id ASC LIMIT 5"),
        ("group_topk",
         "top 5 parts by total quantity used",
         "the 5 most used parts",
         "SELECT p.part_id, p.name, SUM(tp.quantity) AS total_quantity FROM parts p "
         "JOIN ticket_parts tp ON p.part_id = tp.part_id GROUP BY p.part_id, p.name "
         "ORDER BY total_quantity DESC, p.part_id ASC LIMIT 5"),
        ("group_topk",
         "top 3 cities by ticket count",
         "which cities see the most tickets, top 3",
         "SELECT c.city, COUNT(*) AS n FROM customers c "
         "JOIN equipment e ON c.customer_id = e.customer_id "
         "JOIN service_tickets st ON e.equipment_id = st.equipment_id "
         "GROUP BY c.city ORDER BY n DESC, c.city ASC LIMIT 3"),
        ("group_topk",
         "top 5 equipment models by ticket count",
         "the 5 models with the most tickets",
         "SELECT e.model, COUNT(*) AS n FROM equipment e "
         "JOIN service_tickets st ON e.equipment_id = st.equipment_id "
         "GROUP BY e.model ORDER BY n DESC, e.model ASC LIMIT 5"),

        # ---- date range -------------------------------------------------------------
        ("date_range",
         f"tickets opened between {Q_DATE_FROM} and {Q_DATE_TO}",
         f"tickets raised from {Q_DATE_FROM} to {Q_DATE_TO}",
         "SELECT ticket_id, opened_at FROM service_tickets "
         f"WHERE opened_at >= '{Q_DATE_FROM}' AND opened_at <= '{Q_DATE_TO}'"),
        ("date_range",
         "parts used on tickets closed in the last quarter of 2025",
         "parts consumed on tickets closed in Q4 2025",
         "SELECT DISTINCT p.part_id, p.name FROM parts p "
         "JOIN ticket_parts tp ON p.part_id = tp.part_id "
         "JOIN service_tickets st ON tp.ticket_id = st.ticket_id "
         "WHERE st.closed_at >= '2025-10-01' AND st.closed_at < '2026-01-01'"),
        ("date_range",
         "tickets closed in 2025",
         "which tickets were closed during 2025",
         "SELECT ticket_id, closed_at FROM service_tickets "
         "WHERE closed_at >= '2025-01-01' AND closed_at < '2026-01-01'"),
        ("date_range",
         f"equipment installed before {Q_INSTALL_BEFORE}",
         f"machines installed prior to {Q_INSTALL_BEFORE}",
         "SELECT equipment_id, model, install_date FROM equipment "
         f"WHERE install_date < '{Q_INSTALL_BEFORE}'"),
        ("date_range",
         "tickets opened in the first half of 2026",
         "tickets raised in H1 2026",
         "SELECT ticket_id, opened_at FROM service_tickets "
         "WHERE opened_at >= '2026-01-01' AND opened_at < '2026-07-01'"),

        # ---- negation ---------------------------------------------------------------
        ("negation",
         "customers with no tickets in 2026",
         "which customers did not raise a ticket in 2026",
         "SELECT c.customer_id, c.name FROM customers c WHERE NOT EXISTS ("
         "SELECT 1 FROM equipment e JOIN service_tickets st ON e.equipment_id = st.equipment_id "
         "WHERE e.customer_id = c.customer_id "
         "AND st.opened_at >= '2026-01-01' AND st.opened_at < '2027-01-01')"),
        ("negation",
         "technicians who have not closed any tickets",
         "which technicians never closed a ticket",
         "SELECT t.technician_id, t.name FROM technicians t WHERE NOT EXISTS ("
         "SELECT 1 FROM service_tickets st WHERE st.technician_id = t.technician_id "
         "AND st.status = 'closed')"),
        ("negation",
         "equipment with no service tickets",
         "machines that never needed service",
         "SELECT e.equipment_id, e.model FROM equipment e WHERE NOT EXISTS ("
         "SELECT 1 FROM service_tickets st WHERE st.equipment_id = e.equipment_id)"),
        ("negation",
         "parts never used on any ticket",
         "which parts have never been consumed",
         "SELECT p.part_id, p.name FROM parts p WHERE NOT EXISTS ("
         "SELECT 1 FROM ticket_parts tp WHERE tp.part_id = p.part_id)"),
        ("negation",
         "customers without equipment",
         "which customers own nothing",
         "SELECT c.customer_id, c.name FROM customers c WHERE NOT EXISTS ("
         "SELECT 1 FROM equipment e WHERE e.customer_id = c.customer_id)"),
    ]


# Vague on purpose: no table, column or value in the schema is named. The correct behaviour
# is to ask, not to answer. A system that scores well here is one that knows what it cannot do.
_AMBIGUOUS = [
    ("which ones are bad", "show me the bad ones"),
    ("what about last month", "and last month"),
    ("show me the important stuff", "give me the important things"),
    ("how is it going", "how are things looking"),
    ("the usual report", "run the usual one"),
]


def build_questions() -> list[dict]:
    out: list[dict] = []
    n = 0
    for category, question, paraphrase, gold in _templates():
        for text, is_para in ((question, False), (paraphrase, True)):
            n += 1
            out.append({
                "id": f"q-{n:03d}", "category": category, "question": text,
                "is_paraphrase": is_para, "gold_sql": gold, "expected": "sql",
            })
    for question, paraphrase in _AMBIGUOUS:
        for text, is_para in ((question, False), (paraphrase, True)):
            n += 1
            out.append({
                "id": f"q-{n:03d}", "category": "ambiguous", "question": text,
                "is_paraphrase": is_para, "gold_sql": None, "expected": "clarify",
            })
    return out
