"""The market lane: watch Polymarket markets, forecast them, keep score, and guide the owner.

    uv run python markets.py                 look at the markets that are due, forecast, write alerts
    uv run python markets.py --dry-run       the same, but record and tell nothing
    uv run python markets.py --settle-check  compare outcomes read from Polymarket with known ones

It reads public data only. It has no account on Polymarket and cannot place, change or close
a bet: it tells the owner what it sees and what it would do, and the owner acts.

Everything personal lives in one private folder, which the workflow keeps in the owner's
private market-alerts repository:

    watch.txt      the owner's markets and bets (they edit this)
    ledger.jsonl   every forecast, with the market price at the time and, later, the outcome
    scorecard.md   how the bot has done on the markets that have resolved, and the fake-coin wallets
    bets.jsonl     every practice bet the betting rules made with fake coins (see wallet.py)
    alerts/        what to tell the owner after a run; each file becomes a private issue
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import dotenv

dotenv.load_dotenv()

import autopilot
import bench
import hooks
import models
import tools
import wallet

logger = logging.getLogger(__name__)

PRIVATE = Path(os.getenv("PRIVATE_DIR") or "private")
AUTO_MARKETS = int(os.getenv("AUTO_MARKETS") or 6)  # busiest markets closing in 3 to 90 days, watched unasked
NEAR_MARKETS = int(os.getenv("NEAR_MARKETS") or 6)  # busiest markets closing within 3 days: they grade the bot quickly
PRICE_MOVE = float(os.getenv("PRICE_MOVE") or 0.1)  # a price change this large is worth telling the owner
EDGE = 0.1  # the bot's view differs from the price by this much: worth saying, though not worth acting on
NEARLY_WON = 0.9  # at this price a winning bet has little left to gain, so locking it in is worth a thought
# The bot earns a say on one kind of market only from its own live record there: at least this
# many resolved markets, on which it was closer to the truth than the price by this clear a margin.
EARN_MARKETS, EARN_T = 40, 2.5
UNTESTED = "No watched market has resolved yet. In tests on past questions the market price was right more often than the bot."


def read_watch(text: str) -> tuple[list[tuple[str, dict]], list[str]]:
    """The owner's markets and bets from watch.txt, and a note for each line that could not be read.

    A line is a market link, optionally followed by a bet: `yes` or `no`, `bought=` the price
    paid, and optionally `target=` and `stop=`, the prices at which to be told to sell. Prices
    are those of the side held, written as 0.32, 32 or 32%.
    """
    entries, problems = [], []
    for number, line in enumerate(text.splitlines(), 1):
        parts = line.split("#")[0].split()
        if not parts:
            continue
        bet: dict = {}
        try:
            for part in parts[1:]:
                name, _, value = part.lower().partition("=")
                if name in ("yes", "no") and not value:
                    bet["side"] = name
                elif name in ("bought", "target", "stop"):
                    price = float(value.rstrip("%"))
                    bet[name] = price / 100 if price > 1 else price
                else:
                    raise ValueError(f"'{part}' is not yes, no, bought=, target= or stop=")
            if bet and not (bet.get("side") and 0 < bet.get("bought", 0) < 1):
                raise ValueError("a bet needs yes or no, and bought= with the price paid")
        except ValueError as problem:
            problems.append(f"Line {number} of watch.txt: {problem}. The market is watched without the bet.")
            bet = {}
        entries.append((parts[0], bet))
    return entries, problems


def watch_list() -> tuple[list[tuple[dict, dict]], list[str]]:
    """Every market to watch with the owner's bet on it, if any: their list, then the busiest markets."""
    path = PRIVATE / "watch.txt"
    entries, problems = read_watch(path.read_text(encoding="utf-8")) if path.exists() else ([], [])
    watch: dict[str, tuple[dict, dict]] = {}
    for link, bet in entries:
        try:
            rows = tools.polymarket(link)
        except Exception as error:  # one bad line must not stop the other markets
            problems.append(f"Could not read the market at {link} ({type(error).__name__}).")
            continue
        if bet and len(rows) != 1:
            problems.append(f"{link} has {len(rows)} open yes/no markets, so its bet was ignored. Link to the one market.")
            bet = {}
        for row in rows:
            watch[row["id"]] = (row, bet)
    try:
        for row in tools.busiest_markets(AUTO_MARKETS, 3, 90) + tools.busiest_markets(NEAR_MARKETS, 0.5, 3):
            watch.setdefault(row["id"], (row, {}))
    except Exception as error:
        problems.append(f"Could not list the busiest markets this time ({type(error).__name__}).")
    return list(watch.values()), problems


def price_move(link: str, price: float, state: dict) -> list[str]:
    """A line for the owner if the price has moved far since they were last told."""
    told = state.get("price_told")
    moved = told is not None and abs(price - told) >= PRICE_MOVE
    if told is None or moved:
        state["price_told"] = price
    return [f"- Price moved from {told:.0%} to {price:.0%}: {link}"] if moved else []


def standing(graded: list[dict]) -> dict[str, dict]:
    """For each kind of market, how the bot has done against the price on the resolved ones.

    `earned` is true once the bot has clearly out-predicted the price on that kind. `t` is the
    usual paired test on squared errors: how many standard errors closer to the truth the bot
    was than the price. This is how the lane adapts: trust follows the record, kind by kind.
    """
    table = {}
    for kind in sorted({bench.kind(row["question"]) for row in graded}):
        part = [row for row in graded if bench.kind(row["question"]) == kind]
        closer = [(row["price"] - row["outcome"]) ** 2 - (row["forecast"] - row["outcome"]) ** 2 for row in part]
        spread = statistics.stdev(closer) if len(closer) > 1 else 0.0
        t = statistics.mean(closer) / (spread / len(closer) ** 0.5) if spread else 0.0
        table[kind] = {
            "markets": len(part), "bot": bench.score(part), "price": bench.score(part, "crowd"),
            "t": t, "earned": len(part) >= EARN_MARKETS and t >= EARN_T,
        }
    return table


def guidance(bet: dict, price: float, forecast: float | None, earned: bool = False) -> tuple[str, str]:
    """What the bot would do with a bet the owner holds: (kind of advice, the sentence).

    The owner's own target and stop come first, then a bet that is nearly won. The bot's own
    forecast is reported but not turned into "sell" unless it has `earned` a say on this kind
    of market (see standing). The default follows 279 resolved questions with a market price:
    the price was right 83% of the time when the two disagreed, and mixing any of the bot's
    forecast into the price made it less accurate.
    """
    yes = bet["side"] == "yes"
    mine = price if yes else 1 - price  # what the owner's side sells for now
    status = f"your {bet['side'].title()} bet bought at {bet['bought']:.0%} is now {mine:.0%} ({mine / bet['bought'] - 1:+.0%})"
    if bet.get("target") and mine >= bet["target"]:
        return "target", f"Cash out: {status}, at your target of {bet['target']:.0%}"
    if bet.get("stop") and mine <= bet["stop"]:
        return "stop", f"Cut the loss: {status}, at your stop of {bet['stop']:.0%}"
    if mine >= NEARLY_WON and mine > bet["bought"]:
        return "cash out", (
            f"Consider cashing out: {status}. Holding to the end can add at most {1 / mine - 1:+.0%}, "
            f"and the whole stake stays at risk"
        )
    if forecast is None:
        return "hold", f"Hold for now: {status}. The bot has not forecast this market yet"
    fair = forecast if yes else 1 - forecast  # the bot's chance that the owner's side wins
    if earned and fair <= mine - EDGE:
        return ("cash out" if mine > bet["bought"] else "cut"), (
            f"Consider {'cashing out' if mine > bet['bought'] else 'cutting the loss'}: {status}. The bot gives your "
            f"side {fair:.0%}, below the price, and on this kind of market it has out-predicted the price"
        )
    if abs(fair - mine) >= EDGE and not earned:
        return "hold", (
            f"Hold: {status}. The bot gives your side {fair:.0%}, but where bot and price have "
            f"disagreed the price has usually been right, so that is no reason to act"
        )
    return "hold", f"Hold: {status}, and the bot gives your side {fair:.0%}"


def _read(name: str) -> list[dict]:
    path = PRIVATE / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def read_ledger() -> list[dict]:
    return _read("ledger.jsonl")


def practise(ledger: list[dict], prices: dict, settled_now: list[dict], at: str) -> tuple[list[str], list[dict], dict]:
    """The fake-coin wallets' turn: report what settled, bet on open forecasts, value what is open.

    A rule bets on a market at the price recorded when the bot first forecast it, and only
    once, so going over every open forecast each time changes nothing for markets already
    considered. Returns the lines for the owner, the new practice bets, and each rule's
    standing. No real money or account is involved anywhere.
    """
    bets = _read("bets.jsonl")
    outcomes = {row["id"]: row["outcome"] for row in ledger if "outcome" in row}
    proven = {name for name, row in wallet.standings(bets, outcomes, {}).items() if row["earned"]}
    lines = []
    for outcome in settled_now:
        mine = [bet for bet in bets if bet["id"] == outcome["id"]]
        if mine:
            won = lambda bet: (bet["side"] == "yes") == bool(outcome["outcome"])  # noqa: E731
            results = [f"{bet['rule']} {'won' if won(bet) else 'lost'} {abs(bet['shares'] - bet['stake'] if won(bet) else bet['stake']):.0f}" for bet in mine]
            lines.append(f"- Practice result (fake coins): {', '.join(results)}: [{mine[0]['question']}]({mine[0]['url']})")
    open_forecasts = [row for key, row in _first_forecasts(ledger).items() if key not in outcomes]
    new = wallet.place(open_forecasts, bets, outcomes, at)
    for market in open_forecasts:
        here = [bet for bet in new if bet["id"] == market["id"]]
        if here:
            bought = "; ".join(f"{bet['rule']} bought {bet['side'].title()} at {bet['paid']:.0%} for {bet['stake']:.0f}" for bet in here)
            lines.append(f"- Practice bet (fake coins): {bought}: [{market['question']}]({market['url']})")
        lines += [
            f"- Worth a look: the rule \"{bet['rule']}\" has made coins in practice, and it has just bought "
            f"{bet['side'].title()} at {bet['paid']:.0%}: [{market['question']}]({market['url']})"
            for bet in here
            if bet["rule"] in proven
        ]
    prices = dict(prices)
    for bet in bets + new:  # an open bet on a market that has left the watch list still needs today's price
        if bet["id"] not in outcomes and bet["id"] not in prices:
            try:
                prices[bet["id"]] = next((row["price"] for row in tools.polymarket(bet["url"]) if row["id"] == bet["id"]), None)
            except Exception:
                prices[bet["id"]] = None  # valued at what it cost
    return lines, new, wallet.standings(bets + new, outcomes, prices)


def _first_forecasts(ledger: list[dict]) -> dict[str, dict]:
    """Each market's first forecast: the fair one to grade, made when the bot knew least."""
    first: dict[str, dict] = {}
    for row in ledger:
        if "forecast" in row:
            first.setdefault(row["id"], row)
    return first


def settle(ledger: list[dict]) -> list[dict]:
    """An outcome record for each forecast market that has resolved since the last look."""
    done = {row["id"] for row in ledger if "outcome" in row}
    found = []
    for key in _first_forecasts(ledger):
        if key in done:
            continue
        try:
            outcome = tools.polymarket_outcome(key.removeprefix("polymarket-"))
        except Exception as error:
            logger.warning(f"Could not check whether {key} has resolved: {type(error).__name__}")
            continue
        if outcome is not None:
            found.append({"id": key, "outcome": outcome, "settled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return found


def _graded(ledger: list[dict]) -> list[dict]:
    outcomes = {row["id"]: row["outcome"] for row in ledger if "outcome" in row}
    return [
        {**row, "crowd": row["price"], "outcome": outcomes[key], "source": "polymarket"}
        for key, row in _first_forecasts(ledger).items()
        if key in outcomes
    ]


def record(ledger: list[dict]) -> str:
    """One sentence on how the bot has done on resolved markets, for the foot of every alert."""
    graded = _graded(ledger)
    if not graded:
        return UNTESTED
    bot, market = bench.score(graded), bench.score(graded, "crowd")
    return (
        f"The bot's record on the {len(graded)} watched markets that have resolved: "
        f"bot right {bot['accuracy']:.0%}, market price right {market['accuracy']:.0%}."
    )


def scorecard(ledger: list[dict], problems: list[str], wallets: dict | None = None) -> str:
    graded, first = _graded(ledger), _first_forecasts(ledger)
    settled = {row["id"] for row in graded}
    lines = ["# Scorecard", "", f"Markets forecast: {len(first)}. Resolved: {len(graded)}.", "", record(ledger)]
    if graded:
        bot, market, bets = bench.score(graded), bench.score(graded, "crowd"), bench.paper(graded)
        lines += [
            "",
            "| | Right | Brier (lower is better) |", "|---|---|---|",
            f"| Bot | {bot['accuracy']:.0%} | {bot['brier']:.3f} |",
            f"| Market price when the bot forecast | {market['accuracy']:.0%} | {market['brier']:.3f} |",
            "",
            f"Paper bets (one unit on the bot's side wherever it differed from the price by 5 points or more): "
            f"{bets['bets']} bets, {bets['won']} won, {bets['return']:+.0%} on the money staked.",
            "", "## By kind of market", "",
            f"The bot earns a say on a kind after {EARN_MARKETS} resolved markets there, if it was clearly closer to the "
            f"truth than the price (a margin of {EARN_T} or more). Until then its view is shown but not acted on.",
            "",
            "| Kind | Resolved | Bot right | Price right | Bot Brier | Price Brier | Margin | Earned a say |", "|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {kind} | {row['markets']} | {row['bot']['accuracy']:.0%} | {row['price']['accuracy']:.0%} | "
            f"{row['bot']['brier']:.3f} | {row['price']['brier']:.3f} | {row['t']:+.1f} | {'yes' if row['earned'] else 'no'} |"
            for kind, row in standing(graded).items()
        ]
        lines += ["", "## Resolved", ""]
        lines += [
            f"- {'Yes' if row['outcome'] else 'No'}: bot {row['forecast']:.0%}, market {row['price']:.0%}. [{row['question']}]({row['url']})"
            for row in graded
        ]
    if wallets:
        lines += [
            "", "## Practice betting with fake coins", "",
            f"Each betting rule started with {wallet.START:,.0f} fake coins and bets {wallet.STAKE:.0%} of its spare coins, by itself, "
            f"on the markets the bot forecasts. No real money is involved. A rule earns a say after {wallet.EARN_BETS} settled "
            f"bets if its gain per bet is clearly above zero (a margin of {wallet.EARN_T} or more); its new bets are then pointed out to you.",
            "",
            "| Rule | Bets | Settled | Won | Gain per coin staked | Margin | Coins now | Earned a say |", "|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {name} | {row['bets']} | {row['settled']} | {row['won']} | {row['per_bet']:+.0%} | {row['t']:+.1f} | "
            f"{row['coins']:,.0f} | {'yes' if row['earned'] else 'no'} |"
            for name, row in wallets.items()
        ]
    lines += ["", "## Still open", ""]
    lines += [
        f"- Bot {row['forecast']:.0%}, market then {row['price']:.0%}, closes {row['closes'][:10]}. [{row['question']}]({row['url']})"
        for key, row in first.items()
        if key not in settled
    ]
    if problems:
        lines += ["", "## Problems", ""] + [f"- {problem}" for problem in problems]
    return "\n".join(lines) + "\n"


def watch(bot: autopilot.Autopilot, tell: bool) -> tuple[list, list[str]]:
    """One pass over the watched markets. Returns the questions and the public summary lines."""
    listed, problems = watch_list()
    questions = [autopilot.as_question(row) for row, _ in listed]
    reports, summary = asyncio.run(autopilot.run(bot, questions, one_shot=False, record=tell))
    made = {autopilot._key(report.question): report.prediction for report in reports if not isinstance(report, BaseException)}
    ledger, now = read_ledger(), datetime.now(timezone.utc)
    table = standing(_graded(ledger))
    fresh, private = [], []
    for (row, bet), question in zip(listed, questions):
        key, link = autopilot._key(question), f"[{row['question']}]({row['url']})"
        earned = table.get(bench.kind(row["question"]), {}).get("earned", False)
        if key in made:
            fresh.append({
                "id": row["id"], "question": row["question"], "url": row["url"], "at": now.isoformat(timespec="seconds"),
                "forecast": made[key], "price": row["price"], "closes": row["market_info_close_datetime"],
                "saw_price": autopilot.ANCHOR,
            })
        state = hooks.load_state(key)
        summary += price_move(link, row["price"], state)
        if bet:  # the owner's bets never reach the public summary
            latest = next((entry["forecast"] for entry in reversed(ledger + fresh) if entry["id"] == row["id"] and "forecast" in entry), None)
            kind, advice = guidance(bet, row["price"], latest, earned)
            if kind != state.get("advice"):
                state["advice"] = kind
                private.append(f"- {advice}: {link}")
        if tell:
            hooks.save_state(key, state)
    first = _first_forecasts(ledger + fresh)
    settled_now = settle(ledger + fresh)
    for outcome in settled_now:
        fresh.append(outcome)
        row = first[outcome["id"]]
        private.append(
            f"- Resolved {'Yes' if outcome['outcome'] else 'No'}: [{row['question']}]({row['url']}). "
            f"The bot said {row['forecast']:.0%}, the market {row['price']:.0%}"
        )
    ledger += fresh
    practice, new_bets, wallets = practise(
        ledger, {row["id"]: row["price"] for row, _ in listed}, settled_now, now.isoformat(timespec="seconds")
    )
    alerts = [line for line in summary if line.startswith(("- Forecast", "- Price"))] + private + practice
    if tell:
        (PRIVATE / "alerts").mkdir(parents=True, exist_ok=True)
        for name, entries in (("ledger.jsonl", fresh), ("bets.jsonl", new_bets)):
            with open(PRIVATE / name, "a", encoding="utf-8") as out:
                out.writelines(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
        (PRIVATE / "scorecard.md").write_text(scorecard(ledger, problems, wallets), encoding="utf-8")
        if alerts:
            coins = ", ".join(f"{name} {row['coins']:,.0f}" for name, row in wallets.items())
            foot = ["", record(ledger), f"Fake-coin wallets (each started with {wallet.START:,.0f}): {coins}."]
            (PRIVATE / "alerts" / f"{now:%Y%m%dT%H%M%SZ}.md").write_text("\n".join(alerts + foot) + "\n", encoding="utf-8")
    return questions, summary


def settle_check(count: int = 25) -> str:
    """Read outcomes from Polymarket for questions whose outcome is known, and count the agreement."""
    rows = [row for row in bench.exam() if row["source"] == "polymarket"][:count]
    agree, lines = 0, []
    for row in rows:
        slug = urlparse(row["url"]).path.rstrip("/").split("/")[-1]
        try:
            read = tools.polymarket_outcome(slug)
        except Exception as error:
            read = type(error).__name__
        agree += read == row["outcome"]
        if read != row["outcome"]:
            lines.append(f"- read {read}, known {row['outcome']}: {row['question'][:90]}")
    return "\n".join([f"### Settle check: outcomes read from Polymarket match the known ones on {agree} of {len(rows)} resolved questions"] + lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="forecast, but record and tell nothing")
    parser.add_argument("--settle-check", action="store_true", help="only compare outcomes read from Polymarket with known ones")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if args.settle_check:
        text = settle_check()
        print(text)
        if os.getenv("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as page:
                page.write(text + "\n")
        return
    roster = models.FORECAST_MODELS + models.UTILITY_MODELS
    if any(name.startswith("nvidia_nim/") for name in roster) and not os.getenv("NVIDIA_NIM_API_KEY"):
        sys.exit("NVIDIA_NIM_API_KEY is missing. Get a free key at https://build.nvidia.com and put it in .env")
    bot = autopilot.Autopilot(
        research_reports_per_question=1,
        predictions_per_research_report=autopilot.SAMPLES,
        publish_reports_to_metaculus=False,  # there is nowhere to post: this lane only reads and tells
        skip_previously_forecasted_questions=False,
        enable_summarize_research=False,
        extra_metadata_in_explanation=True,
    )
    questions, summary = watch(bot, tell=not args.dry_run)
    autopilot.finish(bot, "market watch", questions, summary)


if __name__ == "__main__":
    main()
