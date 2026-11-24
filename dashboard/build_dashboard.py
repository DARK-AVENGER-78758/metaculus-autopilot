"""Build the public dashboard data from the bot's private notes. Stdlib + wallet.py only; no secrets.

Reads the private folder (scorecard.md, daily.jsonl, bets.jsonl, ledger.jsonl) and an optional
runs.json (GitHub run health), and writes site/ for GitHub Pages: data.json, index.html, context.md.
Only fake-coin, no-real-money figures are published. Nothing from the owner's own bet list is read.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import wallet  # noqa: E402  (stdlib-only module)

PRIVATE = Path(os.getenv("PRIVATE_DIR") or "private")
SITE = Path(os.getenv("SITE_DIR") or "site")


def read_jsonl(name: str) -> list[dict]:
    path = PRIVATE / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def grab(pattern: str, text: str, default: str = "") -> str:
    found = re.search(pattern, text)
    return found.group(1).strip() if found else default


def course(row: dict) -> str:
    if row["coins"] >= wallet.GOAL:
        return "reached"
    if row["closed"] < wallet.LEARN_BETS:
        return "too early to say"
    if row["per_bet"] <= 0:
        return "not on course"
    return f"~{math.log(wallet.GOAL / row['coins']) / math.log(1 + wallet.STAKE * row['per_bet']):,.0f} more bets at this rate"


def main() -> None:
    trades, ledger = read_jsonl("bets.jsonl"), read_jsonl("ledger.jsonl")
    daily = read_jsonl("daily.jsonl")
    outcomes = {row["id"]: row["outcome"] for row in ledger if "outcome" in row}
    standings = wallet.standings(trades, outcomes, {})
    scorecard = (PRIVATE / "scorecard.md").read_text(encoding="utf-8") if (PRIVATE / "scorecard.md").exists() else ""

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bets_today = sum(t.get("action") == "buy" and str(t.get("at", ""))[:10] == today for t in trades)
    lessons = [line[2:] for line in scorecard.splitlines() if line.startswith("- **") and "markets:**" in line]
    record = grab(r"bot right (\d+)%, market price right \d+%", scorecard, "")
    price_record = grab(r"bot right \d+%, market price right (\d+)%", scorecard, "")
    graded = grab(r"record on the (\d+) watched markets", scorecard, "0")
    referee = grab(r"(\*\*Referee:\*\*[^\n]*)", scorecard, "")

    runs = []
    runs_file = SITE / "runs.json"
    if runs_file.exists():
        try:
            runs = json.loads(runs_file.read_text(encoding="utf-8"))
        except ValueError:
            runs = []

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "today": today,
        "bets_today": bets_today,
        "target": 10,
        "goal": wallet.GOAL,
        "start": wallet.START,
        "bot_accuracy": int(record) if record else None,
        "price_accuracy": int(price_record) if price_record else None,
        "graded": int(graded),
        "referee": referee,
        "lessons": lessons,
        "wallets": [
            {
                "name": name,
                "coins": round(row["coins"], 1),
                "open": row["open"],
                "closed": row["closed"],
                "won": row["won"],
                "per_bet": round(row["per_bet"], 4),
                "earned": row["earned"],
                "in_profit": row["coins"] > 1.01 * wallet.START and row["closed"] >= wallet.LEARN_BETS,
                "course": course(row),
            }
            for name, row in sorted(standings.items(), key=lambda kv: -kv[1]["coins"])
        ],
        "daily": [
            {"date": d["date"], "bets": d.get("bets", 0), "closed": d.get("closed", 0), "start": d.get("start", False)}
            for d in daily
        ],
        "runs": runs,
    }

    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    shutil.copyfile(HERE / "index.html", SITE / "index.html")
    shutil.copyfile(HERE / "context.md", SITE / "context.md")
    if (PRIVATE / "scorecard.md").exists():
        shutil.copyfile(PRIVATE / "scorecard.md", SITE / "scorecard.md")  # fake-coin figures only
    (SITE / ".nojekyll").write_text("", encoding="utf-8")  # serve files as-is
    print(f"dashboard built: {len(data['wallets'])} wallets, {len(daily)} days, {bets_today} bets today")


if __name__ == "__main__":
    main()
