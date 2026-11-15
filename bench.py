"""Measure the bot before trusting it.

    uv run python bench.py probe             call every model in the pool once and report what works
    uv run python bench.py run --count 12    sit the next 12 exam questions (resumes where it stopped)
    uv run python bench.py report            score everything sat so far

The exam is ForecastBench (github.com/forecastingresearch/forecastbench-datasets, CC BY-SA):
yes/no questions that have since resolved, each with the true outcome and the crowd's forecast
on the day the set was frozen. Every question is replayed as of that day. The bot sees only
news published before it, reads no web pages (they show today's content), and is told the
replayed date. Metaculus hides outcomes from bot accounts, so its own questions cannot be used.

The replay is a little harder than live running, because the page reader and full article text
are switched off. Treat the scores as a floor, and as a way to compare one setup with another.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv
import requests

dotenv.load_dotenv()

import autopilot
import hooks
import models
from forecasting_tools import BinaryQuestion, QuestionState

BENCH = Path("bench")  # downloads, replayed evidence and results; ignored by git
DATASETS = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
EXAM_SET = "2026-10-11"  # the newest set whose questions have had a month to resolve
MARKET_SOURCES = ("metaculus", "polymarket", "kalshi", "manifold", "infer")  # "will X happen" questions
PARALLEL = 3  # exam questions sat at the same time

PROMPT = (
    "A fair coin is flipped three times. What is the probability of exactly two heads? "
    'Show brief working, then end with a line in exactly this form: "Probability: ZZ%"'
)


# ---------------------------------------------------------------- probe


async def probe_model(model) -> dict:
    """One real call through the same path the bot uses. Never raises."""
    pool = models.Pool("probe", [model])
    had_settings = sorted(setting for setting in models.OPTIONAL_SETTINGS if setting in model.litellm_kwargs)
    started = time.monotonic()
    row = {"model": model.model, "ok": False, "seconds": 0.0, "readable": False, "settings": "", "note": ""}
    try:
        answer, _ = await pool.ask(PROMPT)
        row["ok"] = True
        try:
            row["readable"] = abs(autopilot.read_binary(None, answer) - 0.375) < 0.011
            row["note"] = "" if row["readable"] else f"wrong answer: {answer[-60:]!r}"
        except ValueError as problem:
            row["note"] = f"unreadable: {problem}"
    except Exception as error:
        row["note"] = str(error)[:160]
    row["seconds"] = time.monotonic() - started
    kept = sorted(setting for setting in models.OPTIONAL_SETTINGS if setting in model.litellm_kwargs)
    rejected = sorted(set(had_settings) - set(kept))
    row["settings"] = (
        f"rejected {rejected}" if rejected else "-" if not row["ok"] else "accepted" if had_settings else "none sent"
    )
    return row


async def concurrency(model, sizes=(1, 4, 8)) -> list[tuple[int, float]]:
    """Wall time for N identical calls sent together, straight to the model with no rate limit."""
    timings = []
    for size in sizes:
        started = time.monotonic()
        await asyncio.gather(*[model.invoke(PROMPT) for _ in range(size)], return_exceptions=True)
        timings.append((size, time.monotonic() - started))
    return timings


async def probe() -> None:
    # A short question only shows which models answer. Speed must be timed on a real
    # forecasting prompt: Mistral answered this one in 4 seconds and then timed out on every real one.
    names = list(dict.fromkeys(models.FORECAST_MODELS + models.UTILITY_MODELS))
    rows = [await probe_model(models.member(name)) for name in names]  # one at a time: a clean latency reading
    print("\n| Model | Works | Seconds | Answer readable | Reasoning settings | Note |\n|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['model']} | {'yes' if row['ok'] else 'NO'} | {row['seconds']:.0f} | "
            f"{'yes' if row['readable'] else 'NO'} | {row['settings']} | {row['note']} |"
        )
    working = [row["model"] for row in rows if row["ok"]]
    if working:
        print(f"\nCalls sent together to {working[-1]}:")
        for size, seconds in await concurrency(models.member(working[-1])):
            print(f"- {size} at once: {seconds:.0f} s, about {size / seconds * 60:.0f} a minute")
    print(f"\n{len(working)} of {len(rows)} models answered.")


# ---------------------------------------------------------------- exam


def _download(name: str, url: str) -> dict:
    path = BENCH / name
    if not path.exists():
        BENCH.mkdir(exist_ok=True)
        reply = requests.get(url, timeout=60)
        reply.raise_for_status()
        path.write_bytes(reply.content)
    return json.loads(path.read_text(encoding="utf-8"))


def exam(set_date: str = EXAM_SET) -> list[dict]:
    """Resolved yes/no questions from one ForecastBench set, with the outcome and the crowd's forecast."""
    asked = _download(f"{set_date}-questions.json", f"{DATASETS}/question_sets/{set_date}-llm.json")["questions"]
    resolved = _download(f"{set_date}-resolutions.json", f"{DATASETS}/resolution_sets/{set_date}_resolution_set.json")
    outcomes = {
        (entry["source"], str(entry["id"])): entry["resolved_to"]
        for entry in resolved["resolutions"]
        if not isinstance(entry["id"], list) and entry.get("resolved") and entry["resolved_to"] in (0, 1)
    }
    rows = []
    for question in asked:
        key = (question["source"], str(question["id"]))
        if question["source"] in MARKET_SOURCES and key in outcomes:
            try:
                crowd = float(question["freeze_datetime_value"])
            except (TypeError, ValueError):
                crowd = None
            rows.append({**question, "id": str(question["id"]), "outcome": int(outcomes[key]), "crowd": crowd})
    # A fixed shuffle, so any first N questions mix the sources and repeat runs sit the same ones.
    return sorted(rows, key=lambda row: hashlib.sha1(row["id"].encode()).hexdigest())


def as_question(row: dict) -> BinaryQuestion:
    number = int(hashlib.sha1(row["id"].encode()).hexdigest()[:9], 16)
    present = lambda value: value if value not in (None, "", "N/A") else None  # noqa: E731
    closes = present(row.get("market_info_close_datetime"))
    return BinaryQuestion(
        question_text=row["question"],
        id_of_post=number,
        id_of_question=number,
        page_url=row["url"],
        background_info=present(row.get("background")),
        resolution_criteria=present(row.get("market_info_resolution_criteria")) or row["resolution_criteria"],
        fine_print="",
        open_time=datetime.fromisoformat(row["freeze_datetime"]),
        scheduled_resolution_time=datetime.fromisoformat(closes) if closes else None,
        state=QuestionState.OPEN,
        already_forecasted=False,
    )


def results(label: str) -> list[dict]:
    path = BENCH / f"results-{label}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


async def sit(count: int, label: str, only: str = "", set_date: str = EXAM_SET) -> None:
    """Forecast the next `count` exam questions and append each result as soon as it is in."""
    rows = [row for row in exam(set_date) if re.search(only, row["question"], re.I)]
    done = {row["id"] for row in results(label)}
    todo = [row for row in rows if row["id"] not in done][:count]
    if not todo:
        print(f"All {len(rows)} exam questions have been sat for '{label}'.")
        return
    autopilot.AS_OF = datetime.fromisoformat(todo[0]["freeze_datetime"])  # one freeze date per set
    hooks.STATE_DIR = BENCH / f"state-{label}"  # replayed evidence stays apart from the live bot's
    bot = autopilot.Autopilot(
        research_reports_per_question=1,
        predictions_per_research_report=autopilot.SAMPLES,
        publish_reports_to_metaculus=False,
        enable_summarize_research=False,
    )
    scout = bot.get_llm("scout", "llm")
    slots = asyncio.Semaphore(PARALLEL)

    async def one(row: dict) -> None:
        async with slots:
            started = time.monotonic()
            question = as_question(row)
            state = hooks.load_state(question.id_of_question)
            try:
                await autopilot.gather(scout, question, state, autopilot.AS_OF)
            except Exception as error:
                print(f"evidence failed for {row['question'][:60]}: {error}")
            hooks.save_state(question.id_of_question, state)
            bot.states[question.id_of_question] = state
            report = (await bot.forecast_questions([question], return_exceptions=True))[0]
            if isinstance(report, BaseException):
                print(f"FAILED {row['question'][:70]}: {str(report)[:160]}")
                return
            line = {
                "id": row["id"], "source": row["source"], "question": row["question"], "outcome": row["outcome"],
                "crowd": row["crowd"], "forecast": report.prediction, "evidence": len(state["evidence"]),
                "seconds": round(time.monotonic() - started), "models": models.FORECAST_MODELS,
            }
            with open(BENCH / f"results-{label}.jsonl", "a", encoding="utf-8") as out:
                out.write(json.dumps(line) + "\n")
            verdict = "right" if (report.prediction > 0.5) == bool(row["outcome"]) else "WRONG"
            print(f"{verdict}  bot {report.prediction:.0%}  crowd {row['crowd']}  outcome {row['outcome']}  {row['question'][:80]}")

    await asyncio.gather(*[one(row) for row in todo])
    print(report_text(label))


# ---------------------------------------------------------------- scoring


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% range for a share measured on `total` questions."""
    if not total:
        return 0.0, 1.0
    share = hits / total
    centre = (share + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def score(rows: list[dict], key: str = "forecast") -> dict:
    """Accuracy (right side of 50%) and Brier score (0 is perfect, 0.25 is always saying 50%)."""
    rows = [row for row in rows if row.get(key) is not None]
    hits = sum((row[key] > 0.5) == bool(row["outcome"]) for row in rows)
    low, high = wilson(hits, len(rows))
    return {
        "questions": len(rows),
        "accuracy": hits / len(rows) if rows else 0.0,
        "low": low,
        "high": high,
        "brier": sum((row[key] - row["outcome"]) ** 2 for row in rows) / len(rows) if rows else 0.0,
    }


def report_text(label: str) -> str:
    rows = results(label)
    if not rows:
        return f"No results yet for '{label}'."
    bot, crowd = score(rows), score(rows, "crowd")
    no_share = sum(not row["outcome"] for row in rows) / len(rows)
    lines = [
        f"\n### Exam '{label}': {bot['questions']} resolved yes/no questions",
        "| | Accuracy | 95% range | Brier (lower is better) |",
        "|---|---|---|---|",
        f"| Bot | {bot['accuracy']:.0%} | {bot['low']:.0%} to {bot['high']:.0%} | {bot['brier']:.3f} |",
        f"| Crowd on the freeze date | {crowd['accuracy']:.0%} | {crowd['low']:.0%} to {crowd['high']:.0%} | {crowd['brier']:.3f} |",
        f"| Always answering No | {no_share:.0%} | | {1 - no_share:.3f} |",
        "| Always answering 50% | | | 0.250 |",
        "",
        "Calibration (of the questions given a forecast in each band, how many resolved Yes):",
    ]
    for low, high in ((0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.01)):
        band = [row for row in rows if low <= row["forecast"] < high]
        if band:
            lines.append(f"- forecast {low:.0%} to {min(high, 1):.0%}: {len(band)} questions, {sum(row['outcome'] for row in band) / len(band):.0%} resolved Yes")
    misses = sorted(rows, key=lambda row: -abs(row["forecast"] - row["outcome"]))[:6]
    lines.append("\nLargest misses:")
    for row in misses:
        lines.append(f"- bot {row['forecast']:.0%}, crowd {row['crowd']}, outcome {'Yes' if row['outcome'] else 'No'}: {row['question'][:110]}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["probe", "run", "report"])
    parser.add_argument("--count", type=int, default=12, help="exam questions to sit in this run")
    parser.add_argument("--label", default="baseline", help="a name for this setup, so setups can be compared")
    parser.add_argument("--only", default="", help="sit only questions whose text matches this pattern")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(level=logging.WARNING)
    if args.command == "probe":
        asyncio.run(probe())
    elif args.command == "run":
        asyncio.run(sit(args.count, args.label, args.only))
    else:
        print(report_text(args.label))
