"""Fake-coin wallets: the bot practises betting on real markets, as on a real account but without money.

Each betting rule gets its own wallet of START coins and trades by itself. One coin behaves
like one dollar on Polymarket: a share costs its live price and pays 1 coin if its side wins,
nothing if it loses. What is real: the markets, the live price to buy (the ask) and to sell
(the bid) at the moment of each trade, the outcomes, and the chance of losing a whole stake.
What is not: the coins, and no order is ever sent anywhere.

On every run each rule looks again at every market it could hold, with the bot's latest
forecast and the live price. It sells a position it no longer wants (the bot changed its mind,
or the price moved), may switch to the other side, and buys where it now sees a bet. A rule
that keeps making coins earns the right to be pointed out to the owner: see `earned`.

Known gaps against a real account: prices are seen only when the bot runs, not tick by tick;
a small stake is assumed to fill at the best price; trading fees are not charged.

The owner's goal is to turn START coins into GOAL, by winning bets and in no other way. So
the rules below are fixed, a stake is always STAKE of the spare coins, and `audit` replays
the whole trade log on every run and reports anything that broke a rule. If a rule ever
has to change, RULES_VERSION goes up and every wallet starts again from START.
"""

from __future__ import annotations

import statistics
from datetime import datetime

START = 1000.0  # fake coins each rule starts with
STAKE = 0.02  # share of a wallet's spare coins put on one bet
SPREAD = 0.01  # used only when a market reports no live bid and ask: the shown price, give or take this
GAP = 0.10  # how far the bot must be from the price before a rule treats the two as disagreeing
KEEP = 0.02  # a position bought on a disagreement is kept until the gap has shrunk to this
MAX_OPEN = 25  # positions a rule may hold at once; at 2% each that is half its coins
MIN_LIQUIDITY = 500.0  # dollars resting in a market's order book before a rule will trade it, when the market says
MAX_SPREAD = 0.10  # a wider gap between bid and ask means nobody is there to trade with at a fair price
EARN_BETS, EARN_T = 40, 2.5  # closed bets, and how clearly above zero the gain per bet must be, before a rule earns a say
GOAL = 5000.0  # the owner's target for a wallet; one that gets there stops opening new bets
RULES_VERSION = 2  # stamped on every trade; trades from before the stamp existed were made under version 2
LEARNER = "Follow the best"  # a fifth wallet that copies whichever fixed rule has done best so far
LEARN_BETS = 10  # closed bets a rule needs before the learner will follow it


def _lean(edge: float, held: str | None) -> str | None:
    """Yes when the price is GAP or more below the estimate, No when it is GAP above; a side held is kept down to KEEP."""
    if held == "yes" and edge > KEEP:
        return "yes"
    if held == "no" and edge < -KEEP:
        return "no"
    return "yes" if edge >= GAP else "no" if edge <= -GAP else None


def back_the_bot(forecast: float, price: float, held: str | None = None) -> str | None:
    """Bet the bot's side whenever it disagrees with the price; cash out once the price has come to the bot."""
    return _lean(forecast - price, held)


def fade_the_bot(forecast: float, price: float, held: str | None = None) -> str | None:
    """Bet the price's side whenever the bot disagrees with it. On past questions the price usually won."""
    return _lean(price - forecast, held)


def back_the_favourite(forecast: float, price: float, held: str | None = None) -> str | None:
    """Ignore the bot: buy whichever side is priced 80% to 95%, and sell if it falls under 70%."""
    if held == "yes" and price >= 0.70:
        return "yes"
    if held == "no" and price <= 0.30:
        return "no"
    return "yes" if 0.80 <= price <= 0.95 else "no" if 0.05 <= price <= 0.20 else None


def bot_when_sure(forecast: float, price: float, held: str | None = None) -> str | None:
    """Bet the bot's side only where it is very sure and the price is 5 points behind; sell if the bot stops being sure."""
    if held == "yes" and forecast >= 0.8:
        return "yes"
    if held == "no" and forecast <= 0.2:
        return "no"
    if forecast >= 0.9 and forecast - price >= 0.05:
        return "yes"
    return "no" if forecast <= 0.1 and price - forecast >= 0.05 else None


RULES = {
    "Back the bot": back_the_bot,
    "Fade the bot": fade_the_bot,
    "Back the favourite": back_the_favourite,
    "Bot when very sure": bot_when_sure,
}


WALLETS = [*RULES, LEARNER]


def leader(trades: list[dict], outcomes: dict) -> str | None:
    """The fixed rule the learner follows now: the best gain per closed bet, if it has made coins at all."""
    best, best_gain = None, 0.0
    for name in RULES:
        gains = book(name, trades, outcomes)[2]
        if len(gains) >= LEARN_BETS and statistics.mean(gains) > best_gain:
            best, best_gain = name, statistics.mean(gains)
    return best


def quotes(market: dict) -> tuple[float, float]:
    """(bid, ask) for a Yes share: the live order book when the market reports one, else the shown price give or take SPREAD."""
    bid, ask = market.get("bid"), market.get("ask")
    if bid is None or ask is None or not 0 <= bid <= ask <= 1:
        bid, ask = max(0.0, market["price"] - SPREAD), min(1.0, market["price"] + SPREAD)
    return bid, ask


def buy_at(side: str, bid: float, ask: float) -> float:
    """What a share of a side costs now. A No share costs what a Yes seller would get, taken from 1."""
    return ask if side == "yes" else 1 - bid


def sell_at(side: str, bid: float, ask: float) -> float:
    return bid if side == "yes" else 1 - ask


def _margin(gains: list[float]) -> tuple[float, float]:
    """The average gain per coin staked, and how many standard errors it sits above zero."""
    if not gains:
        return 0.0, 0.0
    spread = statistics.stdev(gains) if len(gains) > 1 else 0.0
    return statistics.mean(gains), statistics.mean(gains) / (spread / len(gains) ** 0.5) if spread else 0.0


def book(rule: str, trades: list[dict], outcomes: dict) -> tuple[float, dict[str, dict], list[float]]:
    """Replay one rule's trades: its spare coins, its open positions by market, and the gain on each closed bet.

    A position still open when its market resolves is paid out: 1 coin a share if its side won.
    """
    coins, held, gains = START, {}, []
    for trade in trades:
        if trade["rule"] != rule:
            continue
        if trade["action"] == "buy":
            coins -= trade["coins"]
            held[trade["id"]] = trade
        else:  # a sale always closes the whole position
            bought = held.pop(trade["id"])
            coins += trade["coins"]
            gains.append(trade["coins"] / bought["coins"] - 1)
    for key in [key for key in held if key in outcomes]:
        bought = held.pop(key)
        paid_out = bought["shares"] * (outcomes[key] if bought["side"] == "yes" else 1 - outcomes[key])
        coins += paid_out
        gains.append(paid_out / bought["coins"] - 1)
    return coins, held, gains


def trade(markets: list[dict], trades: list[dict], outcomes: dict, at: str) -> list[dict]:
    """What each rule does now: sell what it no longer wants, then buy what it now wants.

    Each market carries the bot's latest `forecast` and the live quote. A sale closes the whole
    position at the price a seller gets; a purchase pays the price a buyer pays.
    """
    new = []
    followed = leader(trades, outcomes)
    keep = lambda forecast, price, held=None: held  # noqa: E731  no rule to follow yet: hold what it has, buy nothing
    live = {market["id"]: market for market in markets}
    for name, rule in [*RULES.items(), (LEARNER, RULES[followed] if followed else keep)]:
        coins, held, _ = book(name, trades, outcomes)
        worth = coins + sum(row["value"] for row in positions(trades, outcomes, live) if row["rule"] == name)
        for market in markets:
            key = market["id"]
            if key in outcomes or market.get("forecast") is None:
                continue
            bid, ask = quotes(market)
            if ask - bid > MAX_SPREAD:
                continue  # an empty order book: a real trader could neither buy nor sell here now
            mid = (bid + ask) / 2
            side = held[key]["side"] if key in held else None
            want = rule(market["forecast"], mid, side)
            seen = {
                "rule": name, "id": key, "question": market["question"], "url": market["url"], "at": at,
                "forecast": market["forecast"], "bid": bid, "ask": ask, "rules": RULES_VERSION,
            }
            if side and want != side:
                bought = held.pop(key)
                got = round(bought["shares"] * sell_at(side, bid, ask), 2)
                new.append({**seen, "action": "sell", "side": side, "price": sell_at(side, bid, ask), "shares": bought["shares"], "coins": got})
                coins += got
                side = None
            stake, thin = round(STAKE * coins, 2), (market.get("liquidity") or MIN_LIQUIDITY) < MIN_LIQUIDITY
            buying = want and not side and worth < GOAL  # a wallet that has reached the goal opens nothing new
            if buying and stake >= 1 and len(held) < MAX_OPEN and not thin and 0.01 <= buy_at(want, bid, ask) <= 0.99:
                price = buy_at(want, bid, ask)
                held[key] = {**seen, "action": "buy", "side": want, "price": price, "shares": round(stake / price, 4), "coins": stake}
                new.append(held[key])
                coins -= stake
    return new


def positions(trades: list[dict], outcomes: dict, markets: dict[str, dict]) -> list[dict]:
    """Every open practice position with what it would sell for now. `markets` maps a market id to its live quote."""
    rows = []
    for name in WALLETS:
        for key, bought in book(name, trades, outcomes)[1].items():
            now = None  # unknown: valued at cost
            if key in markets:
                bid, ask = quotes(markets[key])
                if ask - bid > MAX_SPREAD:  # no real bid to sell into: use the last shown price instead
                    bid = ask = markets[key]["price"]
                now = sell_at(bought["side"], bid, ask)
            value = bought["coins"] if now is None else bought["shares"] * now
            rows.append({**bought, "now": now, "value": value, "gain": value / bought["coins"] - 1})
    return rows


def standings(trades: list[dict], outcomes: dict, markets: dict[str, dict]) -> dict[str, dict]:
    """Each rule's wallet: its bets, its results, and its coins with open positions valued at what they would sell for."""
    open_now = positions(trades, outcomes, markets)
    table = {}
    for name in WALLETS:
        coins, held, gains = book(name, trades, outcomes)
        per_bet, t = _margin(gains)
        table[name] = {
            "open": len(held), "closed": len(gains), "won": sum(gain > 0 for gain in gains),
            "spare": coins, "coins": coins + sum(row["value"] for row in open_now if row["rule"] == name),
            "per_bet": per_bet, "t": t, "earned": len(gains) >= EARN_BETS and t >= EARN_T,
        }
    return table


def audit(trades: list[dict], outcomes: dict, known_from: dict | None = None) -> list[str]:
    """The referee: replay every trade and list each one that broke the practice account's rules.

    An empty list means a clean log: every purchase at the live asking price, every sale at
    the live bid, no stake above STAKE of the spare coins, no trade on a market whose result
    was already known, and every wallet's coins adding up to what `book` says it has.

    `known_from` maps a resolved market to the time of the run that learned its result. A
    position still open then is paid out at that moment, exactly as `book` does, so the coins
    and the position count the stake limit is judged against are the ones the wallet really had.
    """
    problems, coins, held = [], {name: START for name in WALLETS}, {name: {} for name in WALLETS}
    results = sorted((when, key) for key, when in (known_from or {}).items() if key in outcomes)

    def pay_out(until: str | None) -> None:
        """Settle every market whose result was known by `until` (all of them, if None)."""
        while results and (until is None or _by(results[0][0], until)):
            _, key = results.pop(0)
            for name in WALLETS:
                bought = held[name].pop(key, None)
                if bought:
                    coins[name] += bought["shares"] * (outcomes[key] if bought["side"] == "yes" else 1 - outcomes[key])

    for number, trade in enumerate(trades, 1):
        name, key = trade.get("rule"), trade.get("id")
        flag = lambda what: problems.append(f"Trade {number} ({name}, {str(trade.get('at'))[:16]}): {what}")  # noqa: E731
        pay_out(str(trade.get("at")))
        if name not in coins:
            flag("made by a wallet that does not exist")
            continue
        if trade.get("rules", RULES_VERSION) != RULES_VERSION:
            flag(f"made under rules version {trade.get('rules')}, not {RULES_VERSION}")
        if known_from and key in known_from and not _by(str(trade.get("at")), known_from[key]):
            flag("made after the market's result was known")
        bid, ask = trade["bid"], trade["ask"]
        if ask - bid > MAX_SPREAD:
            flag("made with no real market to trade against")
        if trade["action"] == "buy":
            if key in held[name]:
                flag("bought a market it already held")
            if abs(trade["price"] - buy_at(trade["side"], bid, ask)) > 1e-9:
                flag("not bought at the live asking price")
            if trade["coins"] > STAKE * coins[name] + 0.01:
                flag(f"staked {trade['coins']:.2f} coins, more than {STAKE:.0%} of its {coins[name]:.2f} spare coins")
            if abs(trade["shares"] * trade["price"] - trade["coins"]) > 0.01:
                flag("its shares, price and coins do not add up")
            if len(held[name]) >= MAX_OPEN:
                flag(f"held more than {MAX_OPEN} positions")
            coins[name] -= trade["coins"]
            held[name][key] = trade
        else:
            bought = held[name].pop(key, None)
            if bought is None:
                flag("sold a position it did not hold")
                continue
            if abs(trade["price"] - sell_at(trade["side"], bid, ask)) > 1e-9:
                flag("not sold at the live bid")
            if abs(trade["coins"] - bought["shares"] * trade["price"]) > 0.01:
                flag("its shares, price and coins do not add up")
            coins[name] += trade["coins"]
    pay_out(None)
    for name in WALLETS if known_from is not None else []:  # the replay and the wallet's own books must agree
        if abs(coins[name] - book(name, trades, outcomes)[0]) > 0.01 * max(1, len(trades)):
            problems.append(f"{name}: the replay ends with {coins[name]:.2f} spare coins, its own books say {book(name, trades, outcomes)[0]:.2f}")
    return problems


def _by(earlier: str, later: str) -> bool:
    """Is the first moment at or before the second?"""
    try:
        return datetime.fromisoformat(earlier) <= datetime.fromisoformat(later)
    except ValueError:
        return earlier <= later


def course(row: dict) -> str:
    """How far a wallet is from the goal at the rate it has actually been winning."""
    if row["coins"] >= GOAL:
        return "reached"
    if row["closed"] < LEARN_BETS:
        return "too early to say"
    if row["per_bet"] <= 0:
        return "not on course"
    import math

    return f"about {math.log(GOAL / row['coins']) / math.log(1 + STAKE * row['per_bet']):,.0f} more bets at this rate"


def backtest(rows: list[dict]) -> dict[str, dict]:
    """Each rule on resolved questions that had a real-money price: one coin a bet, bought SPREAD above the price, held to the end."""
    table = {}
    for name, rule in RULES.items():
        gains = []
        for row in rows:
            side = rule(row["forecast"], row["crowd"]) if 0.02 <= row["crowd"] <= 0.98 else None
            if side:
                paid = min(0.99, (row["crowd"] if side == "yes" else 1 - row["crowd"]) + SPREAD)
                gains.append(((side == "yes") == bool(row["outcome"])) / paid - 1)
        per_bet, t = _margin(gains)
        table[name] = {"bets": len(gains), "won": sum(gain > 0 for gain in gains), "per_bet": per_bet, "t": t}
    return table
