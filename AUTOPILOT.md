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
more every two weeks. The first run works through that backlog, about 1,000 questions in
five hours; the daily run after that sits whatever has newly resolved. The score is on
each run's summary page, and the results file is attached to the run.

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
