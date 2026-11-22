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

Every trade pays the market's own taker fee, from the schedule Polymarket publishes for it,
and an order under the market's minimum size is not placed.

Known gaps against a real account: prices are seen only when the bot runs, not tick by tick,
and a small stake is assumed to fill at the best price in the order book.

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
MAX_DAYS = 14  # a rule only opens a bet on a market closing within this many days, so its coins come back and are bet again
EARN_BETS, EARN_T = 40, 2.5  # closed bets, and how clearly above zero the gain per bet must be, before a rule earns a say
GOAL = 5000.0  # the owner's target for a wallet; one that gets there stops opening new bets
# Stamped on every trade. Version 3 (2026-11-22) followed the first live week: it charges each market's real
# taker fee and minimum order, opens bets only on markets closing within MAX_DAYS (every wallet had filled up
# with months-long positions), and holds "Fade the bot" and "Back the favourite" to the end (their early sales
# lost 28% and 38% a bet, while the same bets held to the end gained). Every wallet restarted from START.
RULES_VERSION = 3
LEARNER = "Follow the best"  # a fifth wallet that copies whichever fixed rule has done best so far
LEARN_BETS = 10  # closed bets a rule needs before the learner will follow it
SELF = "Self-trained"  # a wallet that learns, for each kind of market, which rule has made coins there, and follows it


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
    """Bet the price's side whenever the bot disagrees with it, and hold it to the end. On past questions the price usually won."""
    return held or _lean(price - forecast, None)


def back_the_favourite(forecast: float, price: float, held: str | None = None) -> str | None:
    """Ignore the bot: buy whichever side is priced 80% to 95%, and hold it to the end."""
    return held or ("yes" if 0.80 <= price <= 0.95 else "no" if 0.05 <= price <= 0.20 else None)


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
    # Added 2026-11-22 with a fresh 1,000 coins, as a test of the one pattern the first live week
    # suggested: backing the bot paid on price markets (65 closed bets, +27% a bet, not yet a
    # clear margin) and lost on every other kind. Only bets made from now on can prove it.
    "Bot on price markets": back_the_bot,
}
ONLY = {"Bot on price markets": "price"}  # a wallet that opens bets on one kind of market only


WALLETS = [*RULES, LEARNER, SELF]


def leader(trades: list[dict], outcomes: dict) -> str | None:
    """The fixed rule the learner follows now: the best gain per closed bet, if it has made coins at all."""
    best, best_gain = None, 0.0
    for name in RULES:
        gains = book(name, trades, outcomes)[2]
        if len(gains) >= LEARN_BETS and statistics.mean(gains) > best_gain:
            best, best_gain = name, statistics.mean(gains)
    return best


def lessons(trades: list[dict], outcomes: dict) -> dict[str, dict]:
    """What the record teaches so far: for each kind of market, the fixed rule that has made the most a closed bet there.

    `follow` is true only if that rule has LEARN_BETS closed bets on the kind and is ahead.
    This is recomputed from the whole log on every run, so the self-trained wallet changes
    its mind as results come in, with nobody choosing for it.
    """
    gains: dict[str, dict[str, list[float]]] = {}
    for name in RULES:
        for bought, gain in _replay(name, trades, outcomes)[2]:
            if bought.get("kind"):
                gains.setdefault(bought["kind"], {}).setdefault(name, []).append(gain)
    learned = {}
    for kind, by_rule in gains.items():
        tried = {name: values for name, values in by_rule.items() if len(values) >= LEARN_BETS}
        best = max(tried, key=lambda name: statistics.mean(tried[name]), default=None)
        learned[kind] = {
            "rule": best, "bets": len(tried[best]) if best else max(map(len, by_rule.values())),
            "gain": statistics.mean(tried[best]) if best else 0.0,
            "follow": bool(best) and statistics.mean(tried[best]) > 0,
        }
    return learned


def quotes(market: dict) -> tuple[float, float]:
    """(bid, ask) for a Yes share: the live order book when the market reports one, else the shown price give or take SPREAD."""
    bid, ask = market.get("bid"), market.get("ask")
    if bid is None or ask is None or not 0 <= bid <= ask <= 1:
        bid, ask = max(0.0, market["price"] - SPREAD), min(1.0, market["price"] + SPREAD)
    return bid, ask


def fee(market: dict, price: float) -> float:
    """Polymarket's taker fee on one share traded at this price: nothing on a market that charges none.

    The market publishes its rate and exponent; the fee is rate x (price x (1 - price)) ^ exponent
    a share, so it is largest at 50 cents and, as a share of the stake, heaviest on cheap shares.
    """
    return (market.get("fee_rate") or 0.0) * (price * (1 - price)) ** (market.get("fee_exponent") or 1.0)


def _days_left(market: dict, at: str) -> float | None:
    try:
        return (datetime.fromisoformat(market["market_info_close_datetime"]) - datetime.fromisoformat(at)).total_seconds() / 86400
    except (KeyError, TypeError, ValueError):
        return None  # unknown: such a market is not bet on


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


def _replay(rule: str, trades: list[dict], outcomes: dict) -> tuple[float, dict[str, dict], list[tuple[dict, float]]]:
    """One rule's trades replayed: spare coins, open positions by market, and each closed bet with its gain."""
    coins, held, closed = START, {}, []
    for trade in trades:
        if trade["rule"] != rule:
            continue
        if trade["action"] == "buy":
            coins -= trade["coins"]
            held[trade["id"]] = trade
        else:  # a sale always closes the whole position
            bought = held.pop(trade["id"])
            coins += trade["coins"]
            closed.append((bought, trade["coins"] / bought["coins"] - 1))
    for key in [key for key in held if key in outcomes]:
        bought = held.pop(key)
        paid_out = bought["shares"] * (outcomes[key] if bought["side"] == "yes" else 1 - outcomes[key])
        coins += paid_out
        closed.append((bought, paid_out / bought["coins"] - 1))
    return coins, held, closed


def book(rule: str, trades: list[dict], outcomes: dict) -> tuple[float, dict[str, dict], list[float]]:
    """Replay one rule's trades: its spare coins, its open positions by market, and the gain on each closed bet.

    A position still open when its market resolves is paid out: 1 coin a share if its side won.
    """
    coins, held, closed = _replay(rule, trades, outcomes)
    return coins, held, [gain for _, gain in closed]


def trade(markets: list[dict], trades: list[dict], outcomes: dict, at: str) -> list[dict]:
    """What each rule does now: sell what it no longer wants, then buy what it now wants.

    Each market carries the bot's latest `forecast`, the live quote and its fee schedule. A sale
    closes the whole position at the price a seller gets; a purchase pays the price a buyer
    pays. Both pay the market's taker fee.
    """
    new = []
    followed = leader(trades, outcomes)
    keep = lambda forecast, price, held=None: held  # noqa: E731  no rule to follow yet: hold what it has, buy nothing
    live = {market["id"]: market for market in markets}
    learned = {kind: lesson["rule"] for kind, lesson in lessons(trades, outcomes).items() if lesson["follow"]}
    for name in WALLETS:
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
            # Whose rule this wallet uses here: its own, the leader's, or the one its record has taught it for this kind.
            source = {LEARNER: followed, SELF: learned.get(market.get("kind"))}.get(name, name)
            rule = RULES[source] if source else keep
            want = rule(market["forecast"], mid, side)
            scope = ONLY.get(source)
            if scope and not side and market.get("kind") != scope:
                want = None  # not this wallet's kind of market
            seen = {
                "rule": name, "id": key, "question": market["question"], "url": market["url"], "at": at,
                "forecast": market["forecast"], "bid": bid, "ask": ask, "rules": RULES_VERSION,
                "fee_rate": market.get("fee_rate") or 0.0, "fee_exponent": market.get("fee_exponent") or 1.0,
                "kind": market.get("kind"),
            }
            if side and want != side:
                bought = held.pop(key)
                price = sell_at(side, bid, ask)
                paid = round(bought["shares"] * fee(market, price), 4)
                got = round(bought["shares"] * price - paid, 2)
                new.append({**seen, "action": "sell", "side": side, "price": price, "shares": bought["shares"], "fee": paid, "coins": got})
                coins += got
                side = None
            stake, thin = round(STAKE * coins, 2), (market.get("liquidity") or MIN_LIQUIDITY) < MIN_LIQUIDITY
            days = _days_left(market, at)
            buying = want and not side and worth < GOAL  # a wallet that has reached the goal opens nothing new
            buying = buying and days is not None and days <= MAX_DAYS  # coins must come back in time to be bet again
            if buying and stake >= 1 and len(held) < MAX_OPEN and not thin and 0.01 <= buy_at(want, bid, ask) <= 0.99:
                price = buy_at(want, bid, ask)
                shares = round(stake / (price + fee(market, price)), 4)  # the stake covers the shares and the fee on them
                if shares >= (market.get("min_shares") or 0.0):  # Polymarket takes no smaller order
                    held[key] = {
                        **seen, "action": "buy", "side": want, "price": price, "shares": shares,
                        "fee": round(shares * fee(market, price), 4), "coins": stake,
                    }
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
                now -= fee(markets[key], now)  # selling pays the fee too
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
            if abs(trade["shares"] * trade["price"] + trade.get("fee", 0.0) - trade["coins"]) > 0.01:
                flag("its shares, price, fee and coins do not add up")
            if abs(trade.get("fee", 0.0) - trade["shares"] * fee(trade, trade["price"])) > 0.01:
                flag("did not pay the market's published fee")
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
            if abs(trade["coins"] - (bought["shares"] * trade["price"] - trade.get("fee", 0.0))) > 0.01:
                flag("its shares, price, fee and coins do not add up")
            if abs(trade.get("fee", 0.0) - bought["shares"] * fee(trade, trade["price"])) > 0.01:
                flag("did not pay the market's published fee")
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
            if name in ONLY and row.get("kind") != ONLY[name]:
                continue
            side = rule(row["forecast"], row["crowd"]) if 0.02 <= row["crowd"] <= 0.98 else None
            if side:
                paid = min(0.99, (row["crowd"] if side == "yes" else 1 - row["crowd"]) + SPREAD)
                gains.append(((side == "yes") == bool(row["outcome"])) / paid - 1)
        per_bet, t = _margin(gains)
        table[name] = {"bets": len(gains), "won": sum(gain > 0 for gain in gains), "per_bet": per_bet, "t": t}
    return table
