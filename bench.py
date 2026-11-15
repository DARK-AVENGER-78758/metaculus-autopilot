"""Measure the bot before trusting it.

    uv run python bench.py probe             call every model in the pool once and report what works
    uv run python bench.py run --count 12    sit the next 12 exam questions (resumes where it stopped)
    uv run python bench.py report            score everything sat so far
    uv run python bench.py tune              choose an answering rule on half the results, test it on the rest
    Add --set all to use every question set, or --set 2026-09-27 for one.

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
import functools
import hashlib
import json
import logging
import math
import os
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
import tools

BENCH = Path("bench")  # downloads, replayed evidence and results; ignored by git
DATASETS = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
EXAM_SET = "2026-10-11"  # the set the confidence cut-off was chosen on
EXAM_SINCE = "2026-06-21"  # the first set frozen after the exam model (Nemotron 3 Ultra, June 2026) was released
MARKET_SOURCES = ("metaculus", "polymarket", "kalshi", "manifold", "infer")  # "will X happen" questions
PARALLEL = int(os.getenv("EXAM_PARALLEL") or 3)  # exam questions sat at the same time
# The bot answers only when its own forecast is outside this middle band. Inside it, the bot
# did no better than saying 50% (Brier 0.265 against 0.250 on the first 90 questions).
CONFIDENT_BELOW, CONFIDENT_ABOVE = 0.25, 0.75

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


def results(label: str) -> list[dict]:
    path = BENCH / f"results-{label}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def exam_sets() -> list[str]:
    """Every question set the exam model cannot have seen in training, oldest first."""
    token = os.getenv("GITHUB_TOKEN")  # GitHub's shared runners hit the anonymous limit without it
    listing = requests.get(
        "https://api.github.com/repos/forecastingresearch/forecastbench-datasets/contents/datasets/resolution_sets",
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=30,
    )
    listing.raise_for_status()
    names = [entry["name"] for entry in listing.json()]
    return sorted(name[:10] for name in names if name.endswith("_resolution_set.json") and name[:10] >= EXAM_SINCE)


async def sit(count: int, label: str, only: str = "", sets: list[str] | None = None) -> None:
    """Forecast up to `count` exam questions not yet sat, appending each result as soon as it is in.

    A question that appears in several sets is sat once, so every result is a different question.
    """
    sat = {row["id"] for row in results(label)}
    for set_date in sets or [EXAM_SET]:
        rows = [row for row in exam(set_date) if row["id"] not in sat and re.search(only, row["question"], re.I)]
        todo = rows[:count]
        if not todo:
            continue
        await _sit_set(todo, label, set_date)
        sat |= {row["id"] for row in todo}
        count -= len(todo)
        if count <= 0:
            break
    print(report_text(label))


async def _sit_set(todo: list[dict], label: str, set_date: str) -> None:
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
            question = autopilot.as_question(row)
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
                "id": row["id"], "set": set_date, "source": row["source"], "question": row["question"],
                "outcome": row["outcome"], "crowd": row["crowd"], "forecast": report.prediction,
                "evidence": len(state["evidence"]), "seconds": round(time.monotonic() - started),
                "models": models.FORECAST_MODELS, "samples": bot.samples.get(question.id_of_question),
            }
            with open(BENCH / f"results-{label}.jsonl", "a", encoding="utf-8") as out:
                out.write(json.dumps(line) + "\n")
            verdict = "right" if (report.prediction > 0.5) == bool(row["outcome"]) else "WRONG"
            print(f"{verdict}  bot {report.prediction:.0%}  crowd {row['crowd']}  outcome {row['outcome']}  {row['question'][:80]}")

    await asyncio.gather(*[one(row) for row in todo])


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


def confident(rows: list[dict], below: float = CONFIDENT_BELOW, above: float = CONFIDENT_ABOVE) -> list[dict]:
    """The questions the bot would answer: its own forecast is outside the coin-flip band."""
    return [row for row in rows if row["forecast"] <= below or row["forecast"] >= above]


def report_text(label: str, sets: list[str] | None = None) -> str:
    rows = [row for row in results(label) if not sets or row.get("set", EXAM_SET) in sets]
    if not rows:
        return f"No results yet for '{label}'."
    bot, crowd = score(rows), score(rows, "crowd")
    sure = score(confident(rows))
    no_share = sum(not row["outcome"] for row in rows) / len(rows)
    lines = [
        f"\n### Exam '{label}': {bot['questions']} resolved yes/no questions" + (f" from {', '.join(sets)}" if sets else ""),
        f"**Answering only when confident** (its forecast is under {CONFIDENT_BELOW:.0%} or over {CONFIDENT_ABOVE:.0%}): "
        f"**{sure['accuracy']:.0%} right** on {sure['questions']} of {len(rows)} questions "
        f"({sure['questions'] / len(rows):.0%} answered; 95% range {sure['low']:.0%} to {sure['high']:.0%}).",
        "",
        "Answering everything:",
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
    lines.append("\nWhat other cut-offs would give:")
    for edge in (0.35, 0.3, 0.25, 0.2, 0.15, 0.1):
        kept = score(confident(rows, edge, 1 - edge))
        if kept["questions"]:
            lines.append(
                f"- under {edge:.0%} or over {1 - edge:.0%}: {kept['accuracy']:.0%} right on {kept['questions']} "
                f"({kept['questions'] / len(rows):.0%} answered; range {kept['low']:.0%} to {kept['high']:.0%})"
            )
    by_set = sorted({row.get("set", EXAM_SET) for row in rows})
    if len(by_set) > 1:
        lines.append("\nBy question set (all answers, then confident answers only):")
        for name in by_set:
            part = [row for row in rows if row.get("set", EXAM_SET) == name]
            every, sure_part = score(part), score(confident(part))
            lines.append(
                f"- {name}: {every['accuracy']:.0%} on {every['questions']}; "
                f"confident {sure_part['accuracy']:.0%} on {sure_part['questions']}"
            )
    misses = sorted(rows, key=lambda row: -abs(row["forecast"] - row["outcome"]))[:6]
    lines.append("\nLargest misses:")
    for row in misses:
        lines.append(f"- bot {row['forecast']:.0%}, crowd {row['crowd']}, outcome {'Yes' if row['outcome'] else 'No'}: {row['question'][:110]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- tuning

BELOW, ABOVE = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3), (0.7, 0.75, 0.8, 0.85, 0.9, 0.95)  # cut-offs a rule may use
SMALLEST_GROUP = 10  # a kind of question is left out only on the evidence of at least this many answers
# ponytail: a keyword guess at the topic, first match wins. Label with a model if the kinds start to matter.
KINDS = {
    "price": r"\b(btc|eth|bitcoin|ethereum|solana|xrp|crypto\w*|oil|brent|wti|gold|silver|nasdaq|s&p|stocks?|shares?|"
    r"close price|settle|trimmed mean|dip to|market cap)\b",
    "screen": r"\b(rotten tomatoes|box office|oscars?|emmys?|grammys?|billboard|spotify|netflix|album|movie|film|youtube)\b",
    "sport": r"\b(nba|nfl|mlb|nhl|ufc|fifa|uefa|nascar|f1|grand prix|champion\w*|world cup|super bowl|playoffs?|league|"
    r"baseball|football|basketball|soccer|tennis|golf|cricket|olympic\w*|vs\.?|match|game)\b",
    "economy": r"\b(cpi|inflation|gdp|unemployment|jobless|payrolls?|federal funds|fed|ecb|interest rates?|hike|bps|"
    r"recession|earnings|revenue|employees|layoffs|tariffs?)\b",
    "election": r"\b(elect\w*|seats?|senate|house race|governor\w*|mayor\w*|nomin\w*|primary|referendum|ballot|premier|"
    r"parliament\w*|win the)\b",
}


@functools.lru_cache(maxsize=None)
def kind(question: str) -> str:
    return next((name for name, pattern in KINDS.items() if re.search(pattern, question, re.I)), "other")


def labels(row: dict) -> set[str]:
    """What a rule may leave out: where the question came from, and roughly what it is about."""
    return {f"source {row['source']}", f"kind {kind(row['question'])}"}


def answered(rows: list[dict], rule: dict) -> list[dict]:
    return [row for row in confident(rows, rule["below"], rule["above"]) if not labels(row) & set(rule["skip"])]


def choose_rule(rows: list[dict], target: float) -> dict | None:
    """The rule that answers the most of `rows` while being right at least `target` of the time.

    For each pair of cut-offs, the weakest kind of question is left out, one at a time, until
    the target is met or nothing weak is left.
    """
    best, most = None, 0
    for below in BELOW:
        for above in ABOVE:
            rule = {"below": below, "above": above, "skip": []}
            while True:
                kept = answered(rows, rule)
                overall = score(kept)["accuracy"]
                if overall >= target or not kept:
                    break
                parts = {
                    label: score([row for row in kept if label in labels(row)])
                    for label in set().union(*map(labels, kept))
                }
                weak = {
                    label: part["accuracy"]
                    for label, part in parts.items()
                    if part["questions"] >= SMALLEST_GROUP and part["accuracy"] < overall
                }
                if not weak:
                    break
                rule["skip"].append(min(weak, key=weak.get))
            if overall >= target and len(kept) > most:
                best, most = rule, len(kept)
    return best


def _rule_text(rule: dict) -> str:
    skip = f", leaving out {' and '.join(rule['skip'])}" if rule["skip"] else ""
    return f"answer when the forecast is under {rule['below']:.0%} or over {rule['above']:.0%}{skip}"


def tune_text(label: str, target: float = 0.9) -> str:
    """Choose an answering rule on one half of the results and score it on the other half.

    A rule's score on the questions it was chosen on is flattering (92% became 76% on fresh
    questions the first time), so only the other half's score counts. Questions of one
    family stay in one half, or one market's price brackets would sit on both sides.
    """
    rows = results(label)
    halves: tuple[list, list] = ([], [])
    for row in rows:
        halves[hashlib.sha256(tools.family(row).encode()).digest()[0] % 2].append(row)
    lines = [
        f"\n### Tuning '{label}' to be right {target:.0%} of the time: {len(rows)} questions in two halves",
        "Each rule is chosen on one half and tested on the other, which had no part in choosing it.",
    ]
    for chosen_on, fresh in (halves, halves[::-1]):
        rule = choose_rule(chosen_on, target)
        if not rule:
            lines.append(f"- No rule reached {target:.0%} on a half of {len(chosen_on)} questions.")
            continue
        seen, test = score(answered(chosen_on, rule)), score(answered(fresh, rule))
        verdict = "holds" if test["low"] >= target else "reached, but not yet proven" if test["accuracy"] >= target else "does not hold"
        lines.append(
            f"- Rule: {_rule_text(rule)}. Where it was chosen: {seen['accuracy']:.0%} on {seen['questions']} of {len(chosen_on)}. "
            f"**On the other half: {test['accuracy']:.0%} right on {test['questions']} of {len(fresh)}** "
            f"({test['questions'] / max(1, len(fresh)):.0%} answered; 95% range {test['low']:.0%} to {test['high']:.0%}): {verdict}."
        )
    lines.append("The ranges treat questions as independent. Many turn on the same event, so the true ranges are wider.")
    lines += ["", "By source and kind (every answer, then answers outside the 25% to 75% band):", "| | Questions | Right | Confident answers | Right |", "|---|---|---|---|---|"]
    for name in sorted({label for row in rows for label in labels(row)}):
        part = [row for row in rows if name in labels(row)]
        every, sure = score(part), score(confident(part))
        lines.append(f"| {name} | {every['questions']} | {every['accuracy']:.0%} | {sure['questions']} | {sure['accuracy']:.0%} |")
    overall = choose_rule(rows, target)
    if overall:
        lines.append(f"\nChosen on every question (so its own score would flatter it): {_rule_text(overall)}.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["probe", "run", "report", "tune"])
    parser.add_argument("--target", type=float, default=0.9, help="for tune: the share of answers that should be right")
    parser.add_argument("--count", type=int, default=12, help="exam questions to sit in this run")
    parser.add_argument("--label", default="baseline", help="a name for this setup, so setups can be compared")
    parser.add_argument("--only", default="", help="sit only questions whose text matches this pattern")
    parser.add_argument(
        "--set", default=EXAM_SET, help="question set date, several separated by commas, or 'all' for every usable set"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(level=logging.WARNING)
    if args.command == "probe":
        asyncio.run(probe())
    elif args.command == "run":
        asyncio.run(sit(args.count, args.label, args.only, exam_sets() if args.set == "all" else args.set.split(",")))
    elif args.command == "tune":
        print(tune_text(args.label, args.target))
    else:
        print(report_text(args.label, None if args.set in ("all", EXAM_SET) else args.set.split(",")))
