"""Fake-coin wallets: the bot practises betting on real markets without any money.

Each betting rule gets its own wallet of START coins and bets by itself on the markets the bot
forecasts. Nothing here touches a real account or real money. It answers one question: which
way of betting, if any, would have made coins? A rule that keeps making coins earns the right
to be pointed out to the owner; see `earned` in standings().
"""

from __future__ import annotations

import statistics

START = 1000.0  # fake coins each rule starts with
STAKE = 0.02  # share of a wallet's spare coins put on one bet
SPREAD = 0.01  # a paper bet pays this much over the shown price and sells this much under it, as a real trader would
GAP = 0.10  # how far the bot must be from the price before a rule treats the two as disagreeing
EARN_BETS, EARN_T = 40, 2.5  # settled bets, and how clearly above zero the gain per bet must be, before a rule earns a say


def back_the_bot(forecast: float, price: float) -> str | None:
    """Bet the bot's side whenever it disagrees with the price."""
    return "yes" if forecast - price >= GAP else "no" if price - forecast >= GAP else None


def fade_the_bot(forecast: float, price: float) -> str | None:
    """Bet the price's side whenever the bot disagrees with it. On past questions the price usually won."""
    return {"yes": "no", "no": "yes", None: None}[back_the_bot(forecast, price)]


def back_the_favourite(forecast: float, price: float) -> str | None:
    """Ignore the bot: buy whichever side is priced between 80% and 95%."""
    return "yes" if 0.80 <= price <= 0.95 else "no" if 0.05 <= price <= 0.20 else None


def bot_when_sure(forecast: float, price: float) -> str | None:
    """Bet the bot's side only where it is very sure, the zone where it has been right nine times in ten."""
    if forecast >= 0.9 and forecast - price >= 0.05:
        return "yes"
    return "no" if forecast <= 0.1 and price - forecast >= 0.05 else None


RULES = {
    "Back the bot": back_the_bot,
    "Fade the bot": fade_the_bot,
    "Back the favourite": back_the_favourite,
    "Bot when very sure": bot_when_sure,
}


def cost(side: str, price: float) -> float:
    """What one share of a side costs to buy: its shown price plus the spread. A winning share pays 1 coin."""
    return min(0.99, (price if side == "yes" else 1 - price) + SPREAD)


def _won(bet: dict, outcomes: dict) -> bool:
    return (bet["side"] == "yes") == bool(outcomes[bet["id"]])


def _margin(gains: list[float]) -> tuple[float, float]:
    """The average gain per coin staked, and how many standard errors it sits above zero."""
    if not gains:
        return 0.0, 0.0
    spread = statistics.stdev(gains) if len(gains) > 1 else 0.0
    return statistics.mean(gains), statistics.mean(gains) / (spread / len(gains) ** 0.5) if spread else 0.0


def spare(rule: str, bets: list[dict], outcomes: dict) -> float:
    """Coins a rule has free: its start, less everything it has staked, plus what its settled bets paid."""
    mine = [bet for bet in bets if bet["rule"] == rule]
    paid_out = sum(bet["shares"] for bet in mine if bet["id"] in outcomes and _won(bet, outcomes))
    return START - sum(bet["stake"] for bet in mine) + paid_out


def place(markets: list[dict], bets: list[dict], outcomes: dict, at: str) -> list[dict]:
    """The paper bets each rule now makes on freshly forecast markets. A rule bets once on a market."""
    new = []
    for name, pick in RULES.items():
        held = {bet["id"] for bet in bets if bet["rule"] == name}
        coins = spare(name, bets, outcomes)
        for market in markets:
            side = None if market["id"] in held else pick(market["forecast"], market["price"])
            stake = round(STAKE * coins, 2)
            if side and stake >= 1:
                paid = cost(side, market["price"])
                new.append({
                    "rule": name, "id": market["id"], "question": market["question"], "url": market["url"],
                    "side": side, "paid": paid, "stake": stake, "shares": round(stake / paid, 4), "at": at,
                })
                coins -= stake
                held.add(market["id"])
    return new


def standings(bets: list[dict], outcomes: dict, prices: dict) -> dict[str, dict]:
    """Each rule's wallet: bets, results, and its coins with open bets valued at what they would sell for now."""
    table = {}
    for name in RULES:
        mine = [bet for bet in bets if bet["rule"] == name]
        settled = [bet for bet in mine if bet["id"] in outcomes]
        gains = [(bet["shares"] if _won(bet, outcomes) else 0.0) / bet["stake"] - 1 for bet in settled]
        open_value = 0.0
        for bet in mine:
            if bet["id"] not in outcomes:
                price = prices.get(bet["id"])  # unknown: value the bet at what it cost
                selling = bet["paid"] if price is None else max(0.0, (price if bet["side"] == "yes" else 1 - price) - SPREAD)
                open_value += bet["shares"] * selling
        per_bet, t = _margin(gains)
        table[name] = {
            "bets": len(mine), "settled": len(settled), "won": sum(gain > 0 for gain in gains),
            "coins": spare(name, bets, outcomes) + open_value, "per_bet": per_bet, "t": t,
            "earned": len(settled) >= EARN_BETS and t >= EARN_T,
        }
    return table


def backtest(rows: list[dict]) -> dict[str, dict]:
    """Each rule on resolved questions that had a real-money price: one coin per bet, bought at the price plus the spread."""
    table = {}
    for name, pick in RULES.items():
        gains = []
        for row in rows:
            side = pick(row["forecast"], row["crowd"]) if 0.02 <= row["crowd"] <= 0.98 else None
            if side:
                gains.append(((side == "yes") == bool(row["outcome"])) / cost(side, row["crowd"]) - 1)
        per_bet, t = _margin(gains)
        table[name] = {"bets": len(gains), "won": sum(gain > 0 for gain in gains), "per_bet": per_bet, "t": t}
    return table
