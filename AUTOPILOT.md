# Autopilot

An end-to-end forecasting loop built on the official Metaculus template in this repo. It
runs on NVIDIA's free hosted models. Code supplies the facts and catches the failures; the
models only read, judge, and answer in a fixed format. Once deployed, GitHub runs it on a
schedule and nobody has to touch it.

## What happens to each question

| Step | Done by | What it does | If it fails |
|---|---|---|---|
| 1. Intake | code | Sorts questions by closing time and skips any that close in under 10 minutes. | - |
| 2. Scout | utility model | Writes 6 to 10 news searches and 2 to 3 social searches. On a new question it runs twice, the second time after seeing the first headlines. | Searches the question title instead. |
| 3. Hooks | `hooks.py` | Polls Google News, Bing News, Bluesky and Hacker News. Free, no keys. Several questions at once. | A dead feed is skipped. |
| 4. Triage | Jev, or the utility model | Scores each new item 0 to 3 for how much it should move the forecast. | Items are kept as background. |
| 5. Facts | `tools.py` | Clock facts, the text of pages the question links to, the history of a linked Yahoo Finance or FRED series or of a price series the search planner names, and the full text of the top news items. | Each tool fails alone; the briefing is just shorter. |
| 6. Forecast | 3 model families | 6 independent forecasts, 2 per family, using the template's prompts in `main.py`. | The next model in the pool takes over. A retired model is skipped for the run. |
| 7. Read | code | Reads each final answer and checks it: range, option sums, percentiles in order and inside the question's bounds. On date questions it pins the forecast to zero before today, because the template's curve otherwise puts about 10% on dates already past. | One repair by the utility model, then that sample alone is dropped. |
| 8. Aggregate | code | Median for yes/no, clipped to 2-98%. Mean with a 1% floor per option for multiple choice. Averaged curves for numeric and date questions. | Fewer than half the samples surviving fails the question; the next run retries it. |
| 9. Publish | Metaculus API | Posts the forecast and all reasoning, with the model behind each sample named. | The next run retries. |

An optional critic (`CRITIC=rewrite`) has a second model attack each forecast before it is
rewritten. It is off until the benchmark shows it helps.

## Two lanes

**Tournament lane** (`run_bot_on_tournament.yaml`, every 15 minutes). FutureEval and
MiniBench, where the prize money is. Each question is open for about 3 hours and the rules
ask for one forecast per question, so the bot forecasts each once and never updates it.

**Watch lane** (`run_bot_on_metaculus_cup.yaml`, every 3 hours). The long-running questions
listed in `watch_questions.txt`. The forecast is updated when an item scored 2 or higher has
arrived and the last update was at least 12 hours ago, or when the forecast is 7 days old.
The scout rewrites each question's searches every 7 days. To watch another question, add its
URL to `watch_questions.txt`.

## The model pool (`models.py`)

| Role | Models, in order | How they are used |
|---|---|---|
| Forecast | Nemotron 3 Ultra, MiniMax M3.1 Flash, StepFun Step-5, Gemma 4 31B | The first three that answer share the forecasts, with reasoning set high where the model allows. Gemma is the spare. |
| Utility | Nemotron 3 Ultra, MiniMax M3.1 Flash, Gemma 4 31B | The first that answers does the searches, scoring and repairs, with reasoning off. |

The roster was chosen by timing a real 16,000-character forecasting prompt on 2026-11-15:

| Model | Real forecast | Verdict |
|---|---|---|
| Nemotron 3 Ultra | 27-39 s | Forecasts and all utility work (8 s to score 120 headlines) |
| MiniMax M3.1 Flash | 220-250 s | Forecasts |
| StepFun Step-5 | 290-460 s | Forecasts; slow, and only finishes because calls are streamed |
| Gemma 4 31B | about 380 s | Spare |
| Mistral Large 4, GLM 5.3 Flash, DeepSeek V4.1 Flash | never finished | Left out |
| Kimi K2.6, Nemotron 3 Super | not available to the account | Left out |

A short test question is no guide to speed: Mistral answered one in 4 seconds and then
timed out on every real prompt. Re-time the roster with real prompts when it changes.

- Calls are streamed. NVIDIA closes a connection that is silent for 300 seconds, and the
  slower models need longer than that for a full forecast.
- One rate limit covers every call: 18 a minute per process, under half of NVIDIA's roughly
  40 a minute, because both lanes can run at once.
- A 429 waits 15 seconds and retries the same model. A retired or unknown model is skipped
  for the rest of the run. A timeout, server error or empty answer goes to the fastest
  remaining model, and a model that times out twice is benched for the run.
- A model's private reasoning is never used as its answer.
- The posted comment names the sources it read but does not republish their text.
- Questions are forecast one at a time, up to 6 per run (`MAX_FORECASTS_PER_RUN`), at about
  8 minutes each. With three questions running together, MiniMax took twice as long per
  call and one call timed out.
- Change the rosters with the repository variables `FORECAST_MODELS` and `UTILITY_MODELS`
  (comma-separated litellm names). Models from other providers can be mixed in.
- The run fails, and GitHub emails you, when fewer than two forecast models still answer.

## Setup

1. **Metaculus bot account.** Create one at https://www.metaculus.com/futureeval/participate/
   and copy its token.
2. **NVIDIA key.** Sign in at https://build.nvidia.com, open any model, and choose "Get API
   Key". It starts with `nvapi-` and is free.
3. **Local check.** Put both in `.env` (copy `.env.template` if it does not exist), then:

   ```bash
   uv run python test_autopilot.py                              # offline checks, no keys needed
   uv run python bench.py probe                                 # which models work, and how fast
   uv run python autopilot.py --mode test_questions --dry-run   # forecasts, posts nothing
   uv run python autopilot.py --question <Metaculus question URL>   # forecast one question now
   ```

4. **GitHub.** Push this folder to a repository. A public one gets unlimited free Actions
   minutes; a private one gets 2,000 a month, which the 15-minute schedule exceeds.
5. **Secrets.** In Settings → Secrets and variables → Actions, add `METACULUS_TOKEN` and
   `NVIDIA_NIM_API_KEY`.
6. **Test.** Actions → Test Bot → Run workflow. It forecasts the practice questions in the
   [bot testing area](https://www.metaculus.com/tournament/bot-testing-area/). Read the run's
   summary page, then check the forecasts from your bot account.

The two scheduled workflows are switched off in this repository until the bot is cleared
to forecast live questions. To switch one on, remove the `# ` from the two `schedule` lines
at the top of `run_bot_on_tournament.yaml` or `run_bot_on_metaculus_cup.yaml`.

### Asking about a Polymarket market

Actions → **Ask the bot one question** → Run workflow, and paste a Polymarket link (or a
Metaculus one). The answer appears on the run's summary page as the bot's forecast against
the market's price. The same thing from a terminal:
`uv run python autopilot.py --question <Polymarket link>`.

- **Read only.** The bot reads the market's question, rules and price from Polymarket's
  public data. It has no account there, holds no money and cannot place a bet. Nothing is
  posted anywhere, to Polymarket or to Metaculus.
- **Where it runs.** GitHub's servers can reach that public data. Where an internet
  provider blocks Polymarket, the terminal command times out, so use the workflow.
- **The models never see the price.** It is shown beside the forecast, so the forecast is
  the bot's own.
- **Do not read a gap as an edge.** On resolved Polymarket questions in the exam, the
  market's price was right more often than the bot. When the two disagree, the market is
  the better bet.

### Watching Polymarket markets (`markets.py`)

The **Watch Polymarket markets** workflow runs with nobody at the keyboard, about every
40 to 60 minutes. GitHub's own scheduler proved useless for this (two scheduled runs in
a day and a half), so each run starts the next one itself: the next run waits half an
hour in the `pace` environment, which uses no runner, and then works. The schedule
lines remain only to restart the chain if it ever breaks. To stop it, cancel the
waiting run or disable the workflow. It places no bets: it watches, forecasts, keeps
score and tells you what it would do.

| Step | What happens |
|---|---|
| Pick markets | Your own list, the 6 most traded undecided markets closing in 3 to 90 days, and the 6 most traded closing within 3 days. One market per family, so one event's price brackets do not fill the list |
| Set hooks | For each market the scout writes news and social searches, as for any question |
| Pace by end time | The sooner a market closes, the closer it is followed (table below) |
| Forecast | A new market is forecast once, then again when material news arrives or the forecast goes stale |
| Guide you | For each bet you list: cash out, cut the loss, or hold |
| Keep score | Every forecast is logged with the market price at the time. When the market resolves, the bot and the price are both graded |

How closely a market is followed:

| Time left | News checked | Can be forecast again after | Forecast is stale after |
|---|---|---|---|
| 1 day | every hour | 4 hours | 8 hours |
| 3 days | every 3 hours | 12 hours | 1 day |
| 12 days or more | every 12 hours | 12 hours | 4 to 7 days |

Prices are checked on every run whatever the pace.

**Your private repository `market-alerts`** holds everything personal. This repository is
public, so none of it is written or printed here.

| File | What it is |
|---|---|
| `watch.txt` | Your markets and bets. Edit it on GitHub. A bet is a link, your side, and the price paid: `<link> yes bought=0.32 target=0.45 stop=0.25` |
| `scorecard.md` | The bot's record on watched markets that have resolved, next to the market price's record, with paper-bet results |
| `ledger.jsonl` | Every forecast, the price at the time, and the outcome once known |
| `bets.jsonl` | Every practice trade (buy or sell) the betting rules made with fake coins |
| `alerts/` | What the bot had to tell you. A workflow there files each as an issue that mentions you, so GitHub notifies you |

**What the guidance says, and on what basis.** For a bet you hold:

1. Your target is reached: "Cash out".
2. Your stop is hit: "Cut the loss".
3. Your side is priced at 90% or more and you are ahead: "Consider cashing out", because
   little is left to gain and the whole stake is still at risk.
4. Otherwise "Hold", with the bot's own forecast for your side next to the price.

You are told when the advice changes, not every hour. Every alert ends with the bot's live
record against the market price.

The bot's forecast is reported but never turned into "sell", "buy" or "switch sides". That
is what the test on resolved questions says to do. On 279 questions with a market price:

| Test | Result |
|---|---|
| Bot and price on opposite sides of 50% (46 questions) | Price right 83%, bot right 17% |
| Mixing the bot's forecast into the price | Every mix was less accurate than the price alone |
| Paper bets on the bot's side at a 10-point gap | 118 bets, 30 won, 44% of the stake lost |
| Paper bets by kind of question and by source | A loss in every group |

**How the bot earns a say (this is how the lane adapts).** Trust follows the live record,
one kind of market at a time (price, election, sport, economy, screen, other). Once 40
watched markets of a kind have resolved, and the bot was clearly closer to the truth on
them than the price was (a paired test on squared errors, margin 2.5 or more), the bot
has earned a say on that kind. From then on its view can turn a "Hold" into "Consider
cashing out" or "Consider cutting the loss". Nothing has to be switched on: the
scorecard's "By kind of market" table shows how far each kind is from earning it. If the
bot never beats the price on a kind, it never gets a say there.

**Practice betting with fake coins (`wallet.py`).** The bot practises betting as on a real
account, without money. Four betting rules each start with 1,000 fake coins and trade by
themselves. One coin behaves like one dollar on Polymarket.

| | In practice | On a real account |
|---|---|---|
| Markets and outcomes | The real ones | The same |
| Price to buy | The live asking price when the trade is made | The same |
| Price to sell | The live bid | The same |
| A winning share | Pays 1 coin | Pays 1 dollar |
| A losing share | Pays nothing: the stake is lost | The same |
| A change of mind | The position is sold, and the other side may be bought | The same, if you choose to |
| Thin markets | Not traded under 500 dollars of resting orders, or with bid and ask 10 points apart | You could, at a bad price |
| Smallest order | The market's own minimum (often 5 shares) | The same |
| Price updates | Each time the bot runs, about every 40 to 60 minutes | Tick by tick |
| Size | Small stakes, assumed filled at the best price | Large orders move the price |
| Fees | Each market's own taker fee, buying and selling: shares × rate × price × (1 − price), with the rate the market publishes | The same |

On every run each rule looks at every position again, with the bot's latest forecast and
the live price:

| Rule | Buys | Sells |
|---|---|---|
| Back the bot | The bot's side when it is 10 points from the price | When the price has come within 2 points of the bot, or the bot now leans the other way |
| Fade the bot | The price's side in those same cases | Never: held until the market resolves |
| Back the favourite | The side priced 80% to 95%, whatever the bot says | Never: held until the market resolves |
| Bot when very sure | The bot's side under 10% or over 90%, if the price is 5 points behind | When the bot is no longer sure |
| Bot on price markets | As Back the bot, but only on price markets (crypto, oil and the like) | As Back the bot |
| Late favourite | With two days or less to go, the side priced 90% to 98%, whatever the bot says | Never |
| Follow the move | The side the price has moved towards by 10 points since the bot first looked | Never |
| Fade the move | The side it has moved away from, expecting it to come back | Never |

**The goal and the referee.** The owner's goal is to turn 1,000 fake coins into 5,000 by
winning bets, and in no other way. Three things keep that honest:

- **Fixed rules.** A stake is always 2% of spare coins, so no wallet can reach the goal on one
  lucky gamble. If a rule ever has to change, every wallet starts again from 1,000.
- **A referee on every run.** `wallet.audit` replays the whole trade log and reports any trade
  that was not at the live asking price or bid, was over the stake limit, was made after its
  market's result was known, or does not add up. The scorecard shows its verdict.
- **A dated record.** Each run's trades are committed to the private repository with
  GitHub's own timestamp, before the markets resolve.

A fifth wallet, "Follow the best", is the learner: once one of the four rules has 10 closed
bets and is ahead, it copies that rule, and it moves if another rule overtakes. The
scorecard's "Course to the goal" column says how many more bets each wallet would need at
the rate it has actually been winning, or "not on course".

**The first live week (16 to 22 November 2026), under the earlier rules with no fees.**
The bot forecast 138 markets and 93 resolved; the wallets made 676 trades, and the
referee, once its own bug was fixed, found none that broke a rule.

| | Result |
|---|---|
| Bot against the price, 93 resolved markets | Bot right 71%, price right 82% |
| Price markets (47) | Bot 87%, price 87% |
| Every other kind (46: weather, tweet counts, scores) | Bot 56%, price 76% |
| Back the bot | 936 coins from 1,000 |
| Fade the bot | 931 |
| Back the favourite | 958 |
| Bot when very sure | 952 |
| Follow the best | 778 |

Every wallet lost. Three things were learned and built into version 3 of the rules, after
which every wallet restarted from 1,000 (the week-one log is kept as `bets-v2.jsonl`):

- **Fees.** Polymarket charges a taker fee on many markets and publishes each market's
  rate. On a 10-cent share at a 5% rate it is about 4.5% of the stake. The practice
  account now charges it.
- **Wallets clogged.** Every wallet filled its 25 places with positions months from
  resolving and stopped betting. Rules now open bets only on markets closing within 14
  days.
- **Early sales hurt two rules.** Fade the bot's early sales lost 28% a bet and Back the
  favourite's lost 38%, while the same bets held to the end gained. Both now hold to the
  end.

One pattern is worth testing, not yet believing: backing the bot on price markets made
27% a bet over 65 closed bets, before fees and without a clear margin. It has its own
wallet, "Bot on price markets", and only bets made from now on can prove it.

The referee itself was wrong for that week: it forgot to pay a wallet out when its
market resolved, so it reported hundreds of false alarms. It now settles each market at
the run that learned the result and checks its replay against the wallet's own books.

**The daily assessment, and no need to step in.** Once a day, on the first run after
midnight UTC, the bot judges the day before and files one message, "Daily assessment",
in the private repository:

- bets placed (the target is at least 10 a day) and bets closed, with how many made coins;
- whether any wallet is above its 1,000 starting coins, and each wallet's change over the day;
- how the bot's forecasts did against the price on markets that resolved that day;
- what the self-trained wallet currently believes, and the referee's verdict.

If fewer than 10 bets were placed, the bot widens its own intake: it watches four more
near-term markets from then on (up to 30), and narrows again if bets run far above the
target. Nothing needs setting by hand.

Other messages are now rare. A run sends one of its own only when something concerns a
bet you listed (cash out, cut the loss, hold), when a rule that has proved itself places
a bet, or when the referee finds a problem. Forecasts, price moves and practice trades
are in the scorecard and summed up in the daily assessment.

**The newer rules against the first live week.** Before they started betting, the three
rules that ignore the bot were tried on that week's 95 resolved markets, using every price
the bot saw (936 of them) and the real outcomes, one cent above the price and with a fee:

| Rule | Bets | Won | A bet, at a 1% fee rate | At a 5% fee rate |
|---|---|---|---|---|
| Late favourite | 58 | 54 | 1.2% lost | 1.5% lost |
| Follow the move | 45 | 25 | 2.6% lost | 4.2% lost |
| Fade the move | 45 | 20 | 18.1% lost | 19.6% lost |
| Back the favourite, held to the end | 73 | 66 | 0.5% gained | 0.1% gained |

Favourites roughly break even, which is what a well-priced market looks like, and nothing
here shows a profit. The wallets keep betting so that new markets can confirm or overturn
this, and the self-trained wallet stays out wherever no rule is ahead.

**Self-training.** Nothing here retrains the language models; they are hosted and fixed.
What trains itself, on every run and with nobody choosing, is how the bot bets:

- **"Self-trained" wallet.** From every closed practice bet it works out, for each kind of
  market (price, election, sport, economy, screen, other), which fixed rule has made the
  most a bet there. It follows that rule on that kind once the rule has 10 closed bets
  there and is ahead, and it does not bet on a kind where no rule has made coins. The
  scorecard's "What the bot has taught itself" section lists what it currently believes.
- **"Follow the best" wallet.** The same idea across all markets at once.
- **Earned trust.** The bot's own forecast is acted on for a kind of market only after it
  has beaten the price there on 40 resolved markets.
- **The exam.** GitHub's scheduler sits newly resolved questions a few times a day and
  rescored figures appear on each run's page.

All of it restarted with rules version 3, so it is learning again from the first closed
bets under real fees.

A rule stakes 2% of its spare coins a bet and holds at most 25 positions. A market with an
open position is followed until it resolves, so the bot keeps forecasting it and the rule
can react. Every trade is in `bets.jsonl`; the scorecard shows each wallet, every open
position with what it would sell for now, and the latest trades. Each alert ends with the
four balances.

The same rules on 535 resolved questions with a market price (bought once and held to the
end): Back the bot lost 22% per coin, Fade the bot lost 3%, Back the favourite broke even,
Bot when very sure lost 12%. No rule made coins in both halves of those questions for any
kind of question. The live wallets are the fair test, because their trades are made
before anyone knows the outcome. A rule earns a say after 40 closed bets if its gain per
bet is clearly above zero (a margin of 2.5 or more); from then on its new bets are flagged
to you as "Worth a look". Until a rule earns that, treat its trades as practice only.

**Why it does not bet.** Real-money prediction markets are banned where the owner lives,
and Polymarket does not allow trading through a VPN. The evidence also says not to: in a
paper test on 100 resolved Polymarket and Kalshi questions, staking one unit on the bot's
side wherever it differed from the market price by 5 points or more made 70 bets, won 16,
and lost 54% of the money staked. Larger gaps did worse, not better.

Settings (Settings → Secrets and variables → Actions): the secret `ALERTS_KEY` lets a run
reach `market-alerts` and nothing else. The variables `AUTO_MARKETS`, `NEAR_MARKETS` (both
default 6, 0 for none) and `PRICE_MOVE` (default 0.1) are optional.

### Optional extras

| Secret or variable | What it adds |
|---|---|
| `ASKNEWS_API_KEY` | Article summaries on top of the headlines. Free for tournament bots (ask in the AskNews Discord). |
| `TYPESAFE_API_KEY` | Jev does the triage instead of a model, and gives the critic a second opinion. |
| `REQUESTS_PER_MINUTE` | Raise it once `bench.py probe` shows your real limit. |
| `FORECAST_SAMPLES`, `CRITIC`, `RESEARCH_MODEL` | Samples per question (default 6), the critic switch, and an extra web-search model. |
| `REVIEW_BOT_ENABLED=true` | The template's weekly scorecard of resolved forecasts. |

## How good is it

Measured on 2026-11-15 with `bench.py`, on ForecastBench's 11 October 2026 set: 90 yes/no
questions that have since resolved, replayed as of 9 October with only the news published
before that day. "Accuracy" means the forecast was on the right side of 50%. The Brier score
is the squared error, where 0 is perfect and 0.25 is always saying 50%.

| | Accuracy | 95% range | Brier |
|---|---|---|---|
| Bot | 76% | 66% to 83% | 0.157 |
| The crowd on the same day (market price or community forecast) | 84% | 76% to 91% | 0.100 |
| Always answering No | 63% | | 0.367 |

| Slice | Questions | Bot | Crowd |
|---|---|---|---|
| Metaculus questions, the bot's real target | 16 | 94% | 88% |
| Questions the crowd was sure about (under 10% or over 90%) | 39 | 97% | 100% |
| Questions the crowd was unsure about (10% to 90%) | 51 | 59% | 73% |
| Near coin-flips (crowd 30% to 70%) | 31 | 48% | 61% |

What this says:

- **No setup reaches 90% on a mixed set.** A third of these questions were near coin-flips,
  where even the market was right 61% of the time. Accuracy is capped by how uncertain the
  questions are, not by the bot.
- **The bot is well calibrated**, which is what Metaculus scores. Of the questions it put at
  10-30%, 18% happened; at 70-90%, 86% happened. A recalibration fit made it slightly worse,
  so none is applied.
- **Its weak spot was current numbers** (prices, counts, review scores). The price-series
  lookup was added for that: on the 12 price questions accuracy went from 75% to 83% and
  the Brier score from 0.152 to 0.094, against the crowd's 0.086.
- The exam ran one model (Nemotron 3 Ultra, 4 samples) for speed, with the page reader and
  article text off because they would show today's content. The live bot has both and two
  more model families, so treat these numbers as a floor. The three-model setup has not
  been measured.

Metaculus hides outcomes and community forecasts from bot accounts, so its own questions
cannot be used as an exam. To measure a change: `uv run python bench.py run --label <name>`
in batches (it resumes), then `uv run python bench.py report --label <name>`.

### Answering only when confident

In its own middle band (forecasts between 30% and 70%) the bot did no better than saying
50%. So one way to raise accuracy is to count only the questions where its forecast is
under 25% or over 75%, and leave the rest unanswered.

| Questions | Right when confident | Share answered |
|---|---|---|
| The 90 the cut-off was chosen on | 92% (47 of 51) | 57% |
| 34 other questions, from the 27 September set | 76% (16 of 21) | 62% |
| All 124 | 88% (range 78% to 93%) | 58% |

The rule lifts accuracy from 76% into the high 80s. It has not been shown to hold at 90%
on new questions, and 21 held-out answers are too few to say where it settles. The misses
that remain are confident and wrong for one reason: the bot did not know a current number
(a team's win total, a film's review score, who led a statistic). A check that the
evidence states that number was tried and removed, because in a replay almost no evidence
passes it.

The **Exam on resolved questions** workflow settles this at scale. About 1,150 resolved
questions exist in the sets the exam model cannot have seen in training, with roughly 100
more every two weeks. Each run sits about 200 questions before GitHub's time limit stops
it (the first managed 277, not the 1,000 first estimated), so the daily run works through
the backlog over several nights and then sits whatever has newly resolved. The score is
on each run's summary page, and the results file is attached to the run.

On Metaculus itself, leaving a question unanswered scores zero, so this rule is for
measuring accuracy, not something the live bot applies.

### Tuning the rule

`uv run python bench.py tune --label ultra --target 0.9` looks for the rule that answers the
most questions while being right 90% of the time. A rule is a pair of cut-offs, plus kinds
of question to leave out (by source, and by a keyword guess at the topic). It chooses the
rule on one half of the results and scores it on the other half, then swaps the halves.
Only the other-half score counts: a rule always looks good on the questions it was chosen
on. Questions that differ only in a number, such as one market's price brackets, are kept
in the same half.

### After 746 questions (2026-11-22)

The exam has now sat 746 resolved questions: bot 73% right (Brier 0.177), crowd 81%
(0.130). Under 10% or over 90% the bot is right 90% of the time, on 28% of questions. On
the 535 with a market price, mixing any of the bot's forecast into the price still made
it less accurate, and where the two were on opposite sides of 50% the price was right
75% of the time.

### After 384 questions (2026-11-15)

| Measure | Bot | Market or crowd |
|---|---|---|
| Right side of 50%, all 384 questions | 75% (range 70% to 79%) | 84% |
| Brier score, lower is better | 0.168 | 0.115 |
| Right when its forecast is under 25% or over 75% (63% of questions) | 86% | |
| Right when its forecast is under 10% or over 90% (30% of questions) | 93% (range 87% to 96%) | |

- **Where 90% holds.** Only when the bot is very sure. The rule "answer under 10% or over
  95%" was chosen on one half of the questions and scored 98% on 49 answers in the other
  half (range 89% to 100%). The wider rule "under 30% or over 70%" fell from 91% to 78%.
- **Against market prices the bot has no edge.** See the table under "Watching Polymarket
  markets". When the bot is very sure, the price is usually just as sure, so being right
  there earns nothing.
- **Its worst misses** are confident answers to questions about a current standing (who
  leads a statistic, a film's review score) that it could not look up in a replay.

### Next experiment: starting from the price

So far the forecasters never saw the market price, and their own estimates added nothing
to it. With `ANCHOR=price` they are shown the price and told to start from it, moving
only for specific evidence traders may not have acted on. The exam workflow can sit the
same priced questions again this way (inputs: label `anchored`, also_in `ultra`, anchor
on), reusing the same evidence, so the two setups are compared question by question.
The test is whether the anchored forecast beats the price itself. If it does not, the
price remains the best forecast available for these markets.

The exam is scheduled four times a day and sits questions that have a market price
first, so Polymarket and Kalshi results build up fastest.

## Two things to decide before the prize tournament

- **NVIDIA's terms.** The free API is for "internal testing and evaluation purposes, not in
  production", and the terms bar making generated content available to others. Building,
  probing and the practice area fit that. A bot posting forecasts in a prize tournament may
  not. The pool takes any provider, so moving to Metaculus's free model credits (apply with
  the form at https://forms.gle/aQdYMq9Pisrf1v7d8) is a settings change.
- **Market prices.** Prediction-market odds are the strongest single input when a liquid
  market exists. The bot does not use Polymarket or Kalshi prices.

## What still needs a person

- **Once:** the setup above.
- **Each new season** (January, May, September): set the repository variable `TOURNAMENT_IDS`
  to the new tournament's slug plus `minibench`, and update the Cup URL in `watch_questions.txt`.
- **When a model is retired:** the run summary marks it RETIRED. Remove it from the roster.
- **When a run fails:** GitHub emails you. The summary page shows which question failed and why.

## Verified on 2026-11-15

- The whole pipeline ran against Metaculus's practice area with real NVIDIA models and
  posted forecasts and comments for every question type: yes/no, multiple choice, numeric,
  discrete and date. One question takes 7 to 9 minutes, almost all of it waiting on the two
  slower models.
- Eight calls sent at once took no longer than one, so NVIDIA is not throttling parallel calls.
- The Test Bot workflow ran on GitHub's servers: the news feeds answered from there, two
  practice forecasts were posted, and the run took 22 minutes. StepFun Step-5 timed out twice
  in that run and was benched; the other models carried its share.

## Not yet verified

- Yahoo Finance from GitHub's servers. A blocked source is skipped.
- The Jev calls, written from TypeSafe's API reference.
- The keep-alive step that stops GitHub disabling the schedule after 60 quiet days.

## Not built yet

- An exam run of the full three-model setup, and of the critic, to see whether either beats
  the single fast model. Each takes several hours on the free tier.
- Recalibration from the bot's own resolved tournament forecasts, once about 100 have
  resolved. The exam showed none is needed today.
- Lookups for numbers that are not prices (launch counts, review scores, company figures),
  and similar past questions.

## Files

| File | Purpose |
|---|---|
| `autopilot.py` | The loop: scout, triage, answer reading, aggregation, the two lanes, the command line. |
| `models.py` | The model pool: rosters, rate limit, fallback. |
| `tools.py` | The fact tools and the briefing the forecasters read. |
| `hooks.py` | The four feeds, de-duplication, and the per-question state file. |
| `bench.py` | `probe` tests every model in the pool. `run` and `report` sit and score the exam. |
| `watch_questions.txt` | The questions the watch lane keeps up to date. |
| `test_autopilot.py` | 41 offline checks, including failure injection. |
| `main.py` | The upstream template, unchanged. Autopilot subclasses its bot and uses its prompts. |
