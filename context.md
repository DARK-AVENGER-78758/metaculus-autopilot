# Project context

A written snapshot of what this project is and how it got here. It is refreshed when the
dashboard code is redeployed, not live from the chat. For live numbers, see the other tabs.

## What this is

An automated forecasting bot. It reads open prediction questions, forecasts them with free
language models, and (for Polymarket) practices betting on the forecasts with fake coins to
test whether any way of betting would make money. No real money is involved anywhere, and the
bot has no account or wallet on any betting site.

## Who does what

- The forecasts are made by **NVIDIA's free hosted models**, not by Claude. Claude wrote the
  code; those models run inside it. The running bot never calls Claude.
- Everything runs on **GitHub Actions**. Each run starts the next one itself, so the bot keeps
  going on its own. GitHub's schedule and a second workflow are backups that restart it if it
  stalls.
- Claude's role is oversight only: watching, reporting, and fixing a bug if one appears. That
  oversight runs only while a Claude Code session is open; the bot runs with or without it.

## How the practice betting works

Ten betting rules each start with 1,000 fake coins and trade by themselves on the markets the
bot forecasts. They buy at the live asking price, sell at the live bid, and pay each market's
real Polymarket fee. A referee replays every trade on each run to confirm none broke the
rules (fixed 2% stake, live prices, no trade after the result was known). A "self-trained"
wallet learns, per kind of market, which rule has made coins and follows it. The goal set by
the owner is to turn 1,000 fake coins into 5,000 by winning bets only.

## What the evidence says so far

- The bot's forecasts are less accurate than the market price (roughly high-60s% against
  low-80s% on resolved markets).
- Betting on the bot's forecasts lost money across 746 past questions and the first live week.
- "In profit" on the Wallets tab means a wallet is more than 1% above 1,000 coins on at least
  10 closed bets. A wallet needs about 40 closed bets before a result means more than luck.
- No profit has been shown. With ten wallets running, one being ahead early is more likely
  luck than skill, so a lead is only believed once it holds past 40 bets.

## Things that were declined

- Connecting the bot to place real bets on Polymarket (banned where the owner lives; the
  evidence also says it would lose money). The bot only reads public prices.
- A website to covertly track a person's location. That is stalkerware and was refused.

## Honest limits

- This context tab is a snapshot, not a live feed of the chat.
- Free-tier risks that could quietly stall the bot over weeks: GitHub disabling idle scheduled
  workflows after 60 days, the free NVIDIA key being throttled or expiring, and the cached
  state being evicted after about 7 days unused (costs accuracy, not uptime).
