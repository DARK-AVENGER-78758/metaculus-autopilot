"""Offline checks for hooks.py and autopilot.py. No network, no API keys.

Run with:  uv run python test_autopilot.py
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

import autopilot
import bench
import hooks
import markets
import models
import tools
import wallet
from forecasting_tools import (
    BinaryQuestion,
    BinaryReport,
    DateQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOption,
    PredictedOptionList,
    QuestionState,
    ReasonedPrediction,
)

models.REQUESTS_PER_MINUTE = 600000  # the checks must not wait on the real rate limit
models.RATE_LIMIT_WAIT = 0
REAL_READ_PAGE, REAL_SERIES = tools.read_page, tools.series
tools.read_page = lambda url, chars=0: ""  # the checks stay offline; the tool checks below use fakes
tools.series = lambda url, horizon_days, until=None: ""
NOW = datetime.now(timezone.utc)
SCOUT_REPLY = 'Here you go:\n```json\n{"news": ["Asha Rao prime minister", "site:thehindu.com coalition talks"], "social": ["Rao PM bid"]}\n```'
TRIAGE_REPLY = '{"3": [1], "2": [], "1": [2]}'
CRITIQUE = "The draft ignores the base rate. Suggested adjustment: lower."


class FakeLlm(GeneralLlm):
    """Stands in for a real model: records each prompt and returns a scripted reply."""

    def __init__(self, reply) -> None:
        super().__init__(model="fake/model")
        self.reply, self.prompts = reply, []

    async def invoke(self, prompt, system_prompt=None) -> str:
        self.prompts.append(prompt)
        reply = self.reply(prompt) if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        return reply


class Http(Exception):
    """An API error carrying a status code, the way litellm's errors do."""

    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


def named(name: str, reply=None) -> FakeLlm:
    llm = FakeLlm(f"answer from {name}" if reply is None else reply)
    llm.model = name
    return llm


@contextlib.contextmanager
def setting(module, name: str, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def utility_reply(prompt: str) -> str:
    return TRIAGE_REPLY if "You screen incoming news" in prompt else SCOUT_REPLY


def make_question(**overrides) -> BinaryQuestion:
    fields = dict(
        question_text="Will Asha Rao become prime minister before 2030?",
        id_of_post=101, id_of_question=202, page_url="https://www.metaculus.com/questions/101",
        resolution_criteria="Resolves Yes if she is sworn in before 2030.", fine_print="",
        background_info="Coalition talks are under way.", already_forecasted=False, state=QuestionState.OPEN,
    )
    return BinaryQuestion(**{**fields, **overrides})


def make_bot(critic=CRITIQUE, publish=False):
    class TestBot(autopilot.Autopilot):
        async def _run_forecast_on_binary(self, question, research):
            self.research_seen.append(research)
            value = 0.6 if "Second pass" in research else 0.7
            return ReasonedPrediction(prediction_value=value, reasoning=f"reasoning {len(self.research_seen)}")

    bot = TestBot(
        llms={
            "default": FakeLlm("unused"), "critic": FakeLlm(critic), "scout": FakeLlm(utility_reply),
            "parser": FakeLlm("unused"), "summarizer": FakeLlm("summary"), "researcher": "no_research",
        },
        predictions_per_research_report=1,
        publish_reports_to_metaculus=publish,
    )
    bot.research_seen = []
    return bot


def item(title: str, hours_old: float = 1, score: float | None = None, snippet: str = "") -> dict:
    made = hooks._item(title, "https://example.com", "Example", (NOW - timedelta(hours=hours_old)).isoformat(), snippet)
    return made if score is None else {**made, "score": score}


class FakeResponse:
    def __init__(self, content=b"", payload=None):
        self.content, self._payload = content, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_same_headline_gets_same_id():
    assert item("Rao wins vote!")["id"] == item("rao wins  VOTE")["id"]
    assert item("Rao wins vote")["id"] != item("Rao loses vote")["id"]


def test_poll_merges_feeds_drops_old_items_and_survives_a_dead_feed():
    recent = format_datetime(NOW - timedelta(hours=2))
    google = f"""<rss><channel>
        <item><title>Rao to meet president - Reuters</title><link>https://news.google.com/a</link>
          <pubDate>{recent}</pubDate><source url="https://reuters.com">Reuters</source></item>
        <item><title>Ancient story - Old Paper</title><link>x</link>
          <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate><source>Old Paper</source></item>
        </channel></rss>""".encode()
    bing = f"""<rss xmlns:News="https://www.bing.com/news"><channel>
        <item><title>Rao to meet president</title>
          <link>http://www.bing.com/news/apiclick.aspx?ref=x&amp;url=https%3a%2f%2freuters.com%2fa</link>
          <description>The meeting is set for Friday.</description><pubDate>{recent}</pubDate>
          <News:Source>Reuters</News:Source></item>
        </channel></rss>""".encode()
    hacker_news = {"hits": [{"title": "Rao tech policy", "url": None, "objectID": "7", "created_at": NOW.isoformat()}]}

    def fake_get(url, **params):
        if "google" in url:
            assert params["q"].endswith("when:14d")
            return FakeResponse(google)
        if "bing" in url:
            return FakeResponse(bing)
        if "bsky" in url:
            raise ConnectionError("blocked")
        return FakeResponse(payload=hacker_news)

    real_get, hooks._get = hooks._get, fake_get
    try:
        found = hooks.poll([{"source": "news", "query": "Rao"}, {"source": "social", "query": "Rao"}], days=14)
    finally:
        hooks._get = real_get
    by_title = {entry["title"]: entry for entry in found}
    assert set(by_title) == {"Rao to meet president", "Rao tech policy"}, by_title
    merged = by_title["Rao to meet president"]
    assert merged["source"] == "Reuters" and merged["snippet"] == "The meeting is set for Friday."
    assert by_title["Rao tech policy"]["url"] == "https://news.ycombinator.com/item?id=7"


def test_bing_link_is_unwrapped():
    wrapped = "http://www.bing.com/news/apiclick.aspx?ref=x&url=https%3a%2f%2freuters.com%2fa"
    feed = f"<rss><channel><item><title>T</title><link>{wrapped.replace('&', '&amp;')}</link></item></channel></rss>"
    real_get, hooks._get = hooks._get, lambda url, **params: FakeResponse(feed.encode())
    try:
        assert hooks._bing_news("q", 7, None)[0]["url"] == "https://reuters.com/a"
    finally:
        hooks._get = real_get


def test_remember_keeps_relevant_items_ranked_by_tier_then_recency():
    state = hooks.load_state("does-not-exist")
    state["evidence"] = [item("older decisive", hours_old=72, score=3)]
    scored = [item("fresh shift", 1, 2.4), item("fresh decisive", 2, 2.6), item("noise", 1, 0), item("too old", 24 * 60, 3)]
    hooks.remember(state, scored)
    assert [entry["title"] for entry in state["evidence"]] == ["fresh decisive", "older decisive", "fresh shift"]
    assert len(state["seen"]) == 4  # every triaged item is marked seen, relevant or not
    assert "fresh decisive" in hooks.digest(state["evidence"]) and hooks.digest([]) == "No relevant items found."


def test_state_round_trips_and_survives_a_corrupt_file():
    state = hooks.load_state(1)
    state["pending"] = ["headline é"]
    hooks.save_state(1, state)
    assert hooks.load_state(1)["pending"] == ["headline é"]
    (hooks.STATE_DIR / "2.json").write_text("{not json", encoding="utf-8")
    assert hooks.load_state(2)["hooks"] == []


def test_decide():
    def decide(**overrides):
        case = dict(one_shot=False, already_forecasted=True, last_forecast_at=NOW - timedelta(hours=13), pending=[], now=NOW)
        return autopilot.decide(**{**case, **overrides})

    assert decide(already_forecasted=False, last_forecast_at=None) == "new question"
    assert decide(one_shot=True, already_forecasted=False, last_forecast_at=None) == "new question"
    # Bot tournaments: never a second forecast, whatever the news.
    assert decide(one_shot=True, pending=["big news"]) is None
    assert decide(one_shot=True, already_forecasted=False) is None  # our state says we forecast it
    assert decide(pending=["old", "big news"]) == "news: big news"
    assert decide(pending=["big news"], last_forecast_at=NOW - timedelta(hours=2)) is None  # cooling down
    assert decide() is None
    assert decide(last_forecast_at=NOW - timedelta(days=8)) == "forecast went stale"
    assert decide(last_forecast_at=None) == "forecast went stale"  # forecast before, but when is unknown


def test_pool_spreads_samples_over_the_first_three_healthy_models():
    members = [named(name) for name in "abcde"]
    pool = models.Pool("forecast", members, width=3)

    async def six():
        return [await pool.ask("q") for _ in range(6)]

    assert [model for _, model in asyncio.run(six())] == ["a", "b", "c", "a", "b", "c"]
    assert not members[3].prompts and not members[4].prompts  # spares stay idle while the first three work


def test_pool_skips_a_retired_model_for_the_rest_of_the_run():
    retired = named("a", Http(410, "The model 'a' has reached its end of life on 2026-10-15"))
    pool = models.Pool("forecast", [retired, named("b"), named("c"), named("d")], width=3)

    async def four():
        return [await pool.ask("q") for _ in range(4)]

    answered = [model for _, model in asyncio.run(four())]
    assert "a" not in answered and set(answered) == {"b", "c", "d"}
    assert len(retired.prompts) == 1 and "a" in pool.dead  # asked once, never again
    assert any("RETIRED" in line for line in pool.report())


def test_pool_waits_out_a_rate_limit_on_the_same_model():
    replies = iter([Http(429)] * 4 + ["fifth try worked"])
    limited = named("a", lambda prompt: next(replies))
    pool = models.Pool("forecast", [limited, named("b")], width=1)
    assert asyncio.run(pool.ask("q")) == ("fifth try worked", "a")
    assert len(limited.prompts) == 5 and not pool.dead
    throttled = named("a", Http(429))  # still limited after five tries: only then move to the next model
    assert asyncio.run(models.Pool("forecast", [throttled, named("b", "from b")], width=1).ask("q")) == ("from b", "b")
    assert len(throttled.prompts) == 5


def test_pool_drops_settings_a_model_rejects():
    picky = named("a")
    picky.litellm_kwargs["extra_body"] = {"reasoning_effort": "high"}
    picky.reply = lambda prompt: Http(400, "unknown field") if "extra_body" in picky.litellm_kwargs else "fine"
    pool = models.Pool("forecast", [picky, named("b")], width=1)
    assert asyncio.run(pool.ask("q")) == ("fine", "a") and "extra_body" not in picky.litellm_kwargs


def test_pool_never_passes_off_reasoning_as_an_answer():
    assert models.strip_thinking("<think>maybe 80%?</think>Probability: 20%") == "Probability: 20%"
    assert models.strip_thinking("<think>cut off before the answer, maybe 80%") == ""
    thinker = named("a", "<think>cut off before the answer, maybe 80%")
    pool = models.Pool("forecast", [thinker, named("b", "Probability: 20%")], width=1)
    assert asyncio.run(pool.ask("q")) == ("Probability: 20%", "b")


def test_member_streams_and_keeps_only_the_answer():
    def chunk(content=None, reasoning=None, empty=False):
        delta = SimpleNamespace(content=content, reasoning_content=reasoning)
        return SimpleNamespace(choices=[] if empty else [SimpleNamespace(delta=delta)])

    sent = {}

    async def fake_completion(**options):
        sent.update(options)

        async def stream():
            for piece in (chunk(reasoning="maybe 80%?"), chunk(empty=True), chunk("Probability: "), chunk("20%")):
                yield piece

        return stream()

    real, models.litellm.acompletion = models.litellm.acompletion, fake_completion
    try:
        ultra = models.member("nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b")
        assert asyncio.run(ultra.invoke("What are the odds?")) == "Probability: 20%"  # the reasoning text is dropped
    finally:
        models.litellm.acompletion = real
    assert sent["stream"] is True and sent["extra_body"] == {"reasoning_effort": "high"}
    assert sent["messages"] == [{"role": "user", "content": "What are the odds?"}]


def test_a_failed_call_goes_to_the_fastest_model_and_a_slow_model_is_benched():
    slow = named("c", TimeoutError())
    pool = models.Pool("forecast", [named("a"), named("b"), slow], width=3)

    async def nine():
        return [await pool.ask("q") for _ in range(9)]

    answered = [model for _, model in asyncio.run(nine())]
    assert answered[:6] == ["a", "b", "a", "a", "b", "a"]  # c's two turns fell back to a, the first in the roster
    assert "timed out 2 times" in pool.dead["c"] and len(slow.prompts) == 2  # then c is left alone
    assert answered[6:] == ["a", "b", "a"]


def test_pool_fails_loudly_when_every_model_is_gone():
    pool = models.Pool("forecast", [named("a", Http(404, "page not found"))], width=1)
    for expected in ("No forecast model answered", "Every forecast model is unavailable"):
        try:
            asyncio.run(pool.ask("q"))
            raise AssertionError("a pool with no working model must raise")
        except RuntimeError as error:
            assert expected in str(error), error


def test_members_get_reasoning_settings_long_timeouts_and_no_library_retries():
    ultra = models.member("nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b")
    assert ultra.litellm_kwargs["extra_body"] == {"reasoning_effort": "high"}
    assert ultra.litellm_kwargs["timeout"] == 300 and ultra.allowed_tries == 1 and ultra.litellm_kwargs["num_retries"] == 0
    assert "extra_body" not in models.member("nvidia_nim/stepfun-ai/step-5-preview").litellm_kwargs
    bulk = models.member("nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b", forecasting=False)
    assert bulk.litellm_kwargs["extra_body"] == {"reasoning_effort": "none"}  # bulk jobs run without reasoning
    forecast, utility = models.pools()
    assert forecast.width == 3 and len(forecast.members) == 4 and utility.width == 1


LONG_REASONING = "\n".join(f"Point {n}: Rao: looks strong in region {n}" for n in range(12))
PERCENTILES = "\n".join(f"Percentile {level}: {value}" for level, value in [(10, 12), (20, 18), (40, 30), (60, 41), (80, 55), (90, 63)])


def fails(read, question, text) -> bool:
    try:
        read(question, text)
    except Exception:
        return True
    return False


def test_binary_answers_are_read_from_the_last_probability_line():
    question = make_question()
    read = autopilot.read_binary
    assert read(question, "A 30% swing is possible.\nProbability: 15%\n(There is a 5% chance I am wrong.)") == 0.15
    assert read(question, "First pass.\nProbability: 40%\nOn reflection:\n**Probability: 62.5%**") == 0.625
    assert read(question, "Probability: 0.3%") == 0.01 and read(question, "probability: ~99.9 %") == 0.99
    assert fails(read, question, "I think it is quite likely.") and fails(read, question, "Probability: 150%")


def test_multiple_choice_answers_survive_markdown_and_reject_bad_sums():
    question = MultipleChoiceQuestion(
        question_text="Who wins?", id_of_post=5, id_of_question=6, options=["Rao", "Singh", "Someone else"]
    )
    read = autopilot.read_options
    for block in (
        "Rao: 50%\nSingh: 30%\nSomeone else: 20%",
        "- **Option Rao**: 50%\n- **Option Singh**: 30%\n- **Option Someone else**: 20%",
        "Rao: 0.5\nSingh: 0.3\nSomeone else: 0.2\n\nThese sum to 100%.",
    ):
        options = read(question, f"{LONG_REASONING}\n\n{block}").predicted_options
        assert [round(option.probability, 3) for option in options] == [0.5, 0.3, 0.2], block
    assert fails(read, question, f"{LONG_REASONING}\n\nRao: 50%\nSingh: 30%\nSomeone else: 13%")
    assert fails(read, question, f"{LONG_REASONING}\n\nRao: 60%\nSingh: 40%")


def test_numeric_and_date_answers_use_the_final_list_and_reject_invalid_ones():
    numeric = NumericQuestion(
        question_text="How many seats?", id_of_post=1, id_of_question=2, lower_bound=0, upper_bound=100,
        open_upper_bound=True, open_lower_bound=False, unit_of_measure="seats",
    )
    read = autopilot.read_numbers
    draft_then_final = f"First guess:\nPercentile 10: 1\nPercentile 90: 2\n\nI widen it.\n\n{PERCENTILES}"
    decorated = "\n".join(f"- **{line}** seats" for line in PERCENTILES.splitlines())
    for text in (PERCENTILES, draft_then_final, decorated):
        first = read(numeric, text).declared_percentiles[0]
        assert (first.percentile, first.value) == (0.1, 12)
    assert fails(read, numeric, "\n".join(reversed(PERCENTILES.splitlines())))  # decreasing values
    assert fails(read, numeric, PERCENTILES.replace(": ", ": 9000"))  # far outside the question's range
    assert fails(read, numeric, "It will be about 40 seats.")

    midnight = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    date = DateQuestion(
        question_text="When is the vote?", id_of_post=3, id_of_question=4, open_upper_bound=True, open_lower_bound=False,
        lower_bound=midnight - timedelta(days=1500), upper_bound=midnight + timedelta(days=600),
    )

    def dated(offsets) -> str:
        days = [(midnight + timedelta(days=offset)).strftime("%Y-%m-%d") for offset in offsets]
        return "\n".join(f"Percentile {level}: {day}" for level, day in zip((10, 20, 40, 60, 80, 90), days))

    forecast = autopilot.read_dates(date, dated((20, 60, 120, 200, 300, 400)))
    tenth = forecast.declared_percentiles[1]
    assert (tenth.percentile, tenth.value) == (0.1, (midnight + timedelta(days=20)).timestamp())
    # The question's range starts years back, but the event has not happened yet: almost
    # nothing may sit on dates before today (the template's own curve puts about 10% there).
    before_today = max(point.percentile for point in forecast.get_cdf() if point.value <= NOW.timestamp())
    assert before_today < 0.03, before_today
    assert fails(autopilot.read_dates, date, dated((-30, 60, 120, 200, 300, 400)))  # a date already past
    assert fails(autopilot.read_dates, date, "Percentile 10: next spring")
    with_time = dated((20, 60, 120, 200, 300, 400)).replace("\n", "T12:00:00Z\n", 1)
    assert autopilot.read_dates(date, with_time).declared_percentiles[1].value == (midnight + timedelta(days=20, hours=12)).timestamp()
    assert "seats" in autopilot._answer_format(numeric) and str(date.upper_bound.date()) in autopilot._answer_format(date)


def test_an_unreadable_answer_gets_one_repair_then_the_sample_is_dropped():
    bot, question = make_bot(), make_question()
    bot.set_llm(FakeLlm("I think it is quite likely."), "default")
    bot.set_llm(FakeLlm("Probability: 35%"), "parser")
    fixed = asyncio.run(bot._binary_prompt_to_forecast(question, "PROMPT"))
    repair_prompt = bot.get_llm("parser", "llm").prompts[0]
    assert fixed.prediction_value == 0.35 and "restated" in fixed.reasoning and "Model: fake/model" in fixed.reasoning
    assert "quite likely" in repair_prompt and "Probability: ZZ%" in repair_prompt

    bot.set_llm(FakeLlm("Still no number, sorry."), "parser")
    try:
        asyncio.run(bot._binary_prompt_to_forecast(question, "PROMPT"))
        raise AssertionError("a second unreadable answer must raise so the sample is dropped")
    except ValueError:
        pass


def test_forecasts_come_from_several_models_through_the_real_prompts():
    answers = {"a": "Probability: 20%", "b": "Probability: 40%", "c": "<think>hmm</think>Probability: 90%"}
    pool = models.Pool("forecast", [named(name, reply) for name, reply in answers.items()], width=3)
    bot = autopilot.Autopilot(
        llms={"default": pool, "critic": pool, "scout": FakeLlm("unused"), "parser": FakeLlm("unused"),
              "summarizer": FakeLlm("unused"), "researcher": "no_research"},
        predictions_per_research_report=3, enable_summarize_research=False,
    )
    report = asyncio.run(bot.forecast_questions([make_question()]))[0]
    assert report.prediction == 0.4  # the median of 20%, 40% and 90%
    assert all(f"*Model: {name}*" in report.explanation for name in answers)
    assert "Facts computed by code" in report.explanation  # the comment carries the briefing's facts
    assert "Resolves Yes if she is sworn in" in pool.members[0].prompts[0]  # the template's prompt reached the model


def test_aggregation_clips_floors_and_averages():
    bot = make_bot()
    assert asyncio.run(bot._aggregate_predictions([0.99, 0.99, 0.97], make_question())) == 0.98
    assert asyncio.run(bot._aggregate_predictions([0.01, 0.01, 0.05], make_question())) == 0.02
    assert asyncio.run(bot._aggregate_predictions([0.2, 0.9, 0.4], make_question())) == 0.4

    choice = MultipleChoiceQuestion(question_text="Who wins?", id_of_post=5, id_of_question=6, options=["Rao", "Singh", "Other"])
    sure = PredictedOptionList(predicted_options=[
        PredictedOption(option_name=name, probability=chance) for name, chance in zip(choice.options, (0.995, 0.004, 0.001))
    ])
    merged = asyncio.run(bot._aggregate_predictions([sure, sure], choice)).predicted_options
    assert min(option.probability for option in merged) >= 0.01 - 1e-9
    assert abs(sum(option.probability for option in merged) - 1) < 1e-6

    numeric = NumericQuestion(
        question_text="How many seats?", id_of_post=1, id_of_question=2, lower_bound=0, upper_bound=100,
        open_upper_bound=False, open_lower_bound=False,
    )
    low = autopilot.read_numbers(numeric, "\n".join(f"Percentile {p}: {v}" for p, v in [(10, 10), (20, 12), (40, 15), (60, 18), (80, 21), (90, 24)]))
    high = autopilot.read_numbers(numeric, "\n".join(f"Percentile {p}: {v}" for p, v in [(10, 60), (20, 62), (40, 65), (60, 68), (80, 71), (90, 74)]))
    mixed = asyncio.run(bot._aggregate_predictions([low, high], numeric))

    def width(distribution) -> float:
        curve = distribution.get_cdf()
        at = lambda height: next(point.value for point in curve if point.percentile >= height)
        return at(0.9) - at(0.1)

    assert width(mixed) > 2 * max(width(low), width(high))  # two models that disagree give a wide forecast


def test_a_hung_sample_is_dropped_and_the_rest_still_forecast():
    class SlowOnce(autopilot.Autopilot):
        calls = 0

        async def _run_forecast_on_binary(self, question, research):
            type(self).calls += 1
            if type(self).calls == 1:
                await asyncio.sleep(30)
            return ReasonedPrediction(prediction_value=0.7, reasoning="fine")

    bot = SlowOnce(
        llms={"default": FakeLlm("unused"), "critic": FakeLlm("unused"), "scout": FakeLlm("unused"),
              "parser": FakeLlm("unused"), "summarizer": FakeLlm("unused"), "researcher": "no_research"},
        predictions_per_research_report=4, enable_summarize_research=False,
    )
    started = time.monotonic()
    with setting(autopilot, "SAMPLE_TIMEOUT", 0.2):
        report = asyncio.run(bot.forecast_questions([make_question()]))[0]
    assert report.prediction == 0.7 and time.monotonic() - started < 10


class FakePage:
    """Stands in for a streamed web response."""

    def __init__(self, body: bytes = b"", status: int = 200, headers: dict | None = None) -> None:
        self.status_code, self._body, self.raw = status, body, self
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}

    def read(self, limit, decode_content=True) -> bytes:
        return self._body[:limit]

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise ConnectionError(f"HTTP {self.status_code}")

    def close(self) -> None:
        pass


def test_only_public_web_addresses_are_fetched():
    pages = {
        "http://example.org/moved": FakePage(status=302, headers={"location": "http://127.0.0.1/admin"}),
        "http://example.org/big": FakePage(b"x" * 5000),
        "http://example.org/report.pdf": FakePage(b"%PDF", headers={"content-type": "application/pdf"}),
        "http://example.org/gone": FakePage(status=404),
    }
    real_lookup, real_get = tools.socket.getaddrinfo, tools.requests.get
    tools.socket.getaddrinfo = lambda host, port: (
        [(2, 1, 6, "", ("93.184.216.34", port))] if host == "example.org" else real_lookup(host, port)
    )
    tools.requests.get = lambda url, **options: pages[url]
    try:
        for unsafe in (
            "file:///etc/passwd", "ftp://example.org/x", "http://localhost/x", "http://127.0.0.1/x", "http://10.0.0.5/x",
            "http://192.168.1.1/x", "http://169.254.169.254/latest/meta-data", "http://[::1]/x",
        ):
            assert not tools._is_public(unsafe), unsafe
        assert tools._is_public("https://example.org/page")
        try:
            tools.fetch("http://example.org/moved")
            raise AssertionError("a redirect to a private address must be refused")
        except ValueError as error:
            assert "non-public" in str(error)
        assert len(tools.fetch("http://example.org/big", limit=1000)[1]) == 1000  # oversize pages are cut
        for unreadable in ("moved", "report.pdf", "gone"):  # read_page turns every failure into "nothing"
            assert REAL_READ_PAGE(f"http://example.org/{unreadable}") == ""
    finally:
        tools.socket.getaddrinfo, tools.requests.get = real_lookup, real_get


def test_page_text_drops_scripts_and_menus_and_keeps_table_rows():
    page = (
        "<html><head><style>p{color:red}</style><script>var key='secret'</script></head><body>"
        "<nav>Home | About</nav><h1>Flu report</h1><p>Cases rose&nbsp;sharply.</p>"
        "<table><tr><th>Prefecture</th><th>Cases</th></tr><tr><td>Tokyo</td><td>12.4</td></tr></table>"
        "<footer>Contact us</footer></body></html>"
    )
    text = tools.html_to_text(page)
    assert "secret" not in text and "Home" not in text and "Contact" not in text and "color" not in text
    assert "Flu report" in text and "Cases rose sharply." in text and "| Tokyo | 12.4" in text


def test_clock_facts_and_question_links():
    utc = timezone.utc
    question = make_question(
        open_time=datetime(2026, 11, 10, tzinfo=utc), scheduled_resolution_time=datetime(2026, 12, 31, tzinfo=utc),
        resolution_criteria="Per https://www.mhlw.go.jp/flu.html, and https://www.metaculus.com/questions/1/. "
        "(See also https://example.org/a).",
        fine_print="The same page again: https://www.mhlw.go.jp/flu.html and the "
        "[Electoral College](https://en.wikipedia.org/wiki/Electoral_College_\\(United_States\\)).",
    )
    facts = tools.clock(question, datetime(2026, 11, 15, tzinfo=utc))
    assert "Today is 2026-11-15" in facts and "opened on 2026-11-10" in facts
    assert "2026-12-31, 46 days from today" in facts and "3 in 10" in facts and "has not happened yet" in facts
    assert tools.question_links(question) == [
        "https://www.mhlw.go.jp/flu.html",
        "https://example.org/a",
        "https://en.wikipedia.org/wiki/Electoral_College_(United_States)",
    ]


def test_a_polymarket_link_becomes_open_yes_no_questions_with_the_market_price():
    def market(slug, yes, **extra):
        base = {
            "slug": slug, "question": f"Will {slug} win?", "description": "Resolves Yes if they win.",
            "endDate": "2026-12-31T12:00:00Z", "outcomes": '["Yes", "No"]', "outcomePrices": json.dumps([str(yes), str(1 - yes)]),
        }
        return {**base, **extra}

    event = [{"slug": "the-race", "markets": [
        market("rao", 0.2), market("iyer", 0.7), market("old", 0.5, closed=True),
        market("teams", 0.5, outcomes='["Lions", "Tigers"]'),  # not a yes/no market
    ]}]
    asked = []

    def fake_fetch(url):
        asked.append(url)
        return "application/json", json.dumps(event if "/events?" in url else event[0]["markets"][:1])

    with setting(tools, "fetch", fake_fetch):
        rows = tools.polymarket("https://polymarket.com/event/the-race?tid=1")
        one = tools.polymarket("https://polymarket.com/event/the-race/rao")
    assert [(row["question"], row["price"]) for row in rows] == [("Will iyer win?", 0.7), ("Will rao win?", 0.2)]
    assert asked == [f"{tools.GAMMA}/events?slug=the-race", f"{tools.GAMMA}/markets?slug=rao"] and len(one) == 1
    question = autopilot.as_question(rows[0])
    assert question.community_prediction_at_access_time == 0.7 and question.resolution_criteria == "Resolves Yes if they win."
    assert question.scheduled_resolution_time == datetime(2026, 12, 31, 12, tzinfo=timezone.utc)
    try:
        tools.polymarket("https://polymarket.com/profile/someone")
        raise AssertionError("a link that is not a market was accepted")
    except ValueError:
        pass

    # The busiest markets, minus the settled ones and those closing too soon or too far off.
    in_days = lambda days: (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    busy = [
        market("open", 0.4, endDate=in_days(30)), market("settled", 0.97, endDate=in_days(30)),
        market("tomorrow", 0.5, endDate=in_days(1)), market("far", 0.5, endDate=in_days(200)),
        market("undated", 0.5, endDate=None), market("second", 0.6, endDate=in_days(10)),
        market("bracket-50", 0.3, endDate=in_days(30), question="Will oil reach $50?"),
        market("bracket-60", 0.2, endDate=in_days(30), question="Will oil reach $60?"),  # the same market, another number
    ]
    with setting(tools, "fetch", lambda url, limit=0: ("application/json", json.dumps(busy))):
        assert [row["id"] for row in tools.busiest_markets(5)] == ["polymarket-open", "polymarket-second", "polymarket-bracket-50"]
        assert tools.busiest_markets(1)[0]["url"] == "https://polymarket.com/market/open"
        assert [row["id"] for row in tools.busiest_markets(5, 0.5, 3)] == ["polymarket-tomorrow"] and tools.busiest_markets(0) == []
    settled = lambda yes: lambda url: ("application/json", json.dumps([market("done", yes, closed=True)]))  # noqa: E731
    for yes, known in ((1, 1), (0, 0), (0.6, None)):
        with setting(tools, "fetch", settled(yes)):
            assert tools.polymarket_outcome("done") == known
    with setting(tools, "fetch", lambda url: ("application/json", json.dumps([market("open", 1.0)]))):
        assert tools.polymarket_outcome("open") is None  # a price of 100% is not an outcome until the market has closed


def test_market_lane_reads_bets_gives_guidance_keeps_score_and_keeps_bets_private():
    entries, problems = markets.read_watch(
        "# my markets\nhttps://polymarket.com/market/a yes bought=32 target=45% stop=0.25\n\n"
        "https://polymarket.com/market/b\nhttps://polymarket.com/market/c no bought=0.4 sell=0.9\n"
        "https://polymarket.com/market/d target=0.6\n"
    )
    assert entries == [
        ("https://polymarket.com/market/a", {"side": "yes", "bought": 0.32, "target": 0.45, "stop": 0.25}),
        ("https://polymarket.com/market/b", {}), ("https://polymarket.com/market/c", {}), ("https://polymarket.com/market/d", {}),
    ]
    assert len(problems) == 2 and "Line 5" in problems[0] and "'sell=0.9'" in problems[0] and "Line 6" in problems[1]

    bet = {"side": "yes", "bought": 0.32, "target": 0.45, "stop": 0.25}
    assert markets.guidance(bet, 0.46, 0.9)[0] == "target" and "(+44%)" in markets.guidance(bet, 0.46, 0.9)[1]
    assert markets.guidance(bet, 0.24, 0.9)[0] == "stop"  # the owner's own limits come before the bot's view
    # The bot's own view is reported, never turned into "sell": when it disagreed with a price, the price was usually right.
    kind, advice = markets.guidance(bet, 0.40, 0.25)
    assert kind == "hold" and "is now 40% (+25%)" in advice and "The bot gives your side 25%" in advice and "no reason to act" in advice
    assert markets.guidance(bet, 0.30, 0.15)[0] == "hold" and markets.guidance(bet, 0.30, None)[0] == "hold"
    assert markets.guidance(bet, 0.40, 0.45)[1].endswith("and the bot gives your side 45%")
    # Once the bot has earned a say on this kind of market, its view does turn into advice.
    kind, advice = markets.guidance(bet, 0.40, 0.25, earned=True)
    assert kind == "cash out" and "it has out-predicted the price" in advice
    assert markets.guidance(bet, 0.30, 0.15, earned=True)[0] == "cut" and markets.guidance(bet, 0.40, 0.6, earned=True)[0] == "hold"

    # Trust is earned kind by kind from resolved markets: enough of them, and clearly closer to the truth than the price.
    def resolved(count, sure, question="Will the bill pass?"):
        return [
            {"question": question, "price": 0.5, "crowd": 0.5, "outcome": n % 2, "forecast": (sure if n % 2 else 1 - sure) + 0.01 * (n % 3)}
            for n in range(count)
        ]

    table = markets.standing(resolved(40, 0.9) + resolved(39, 0.9, "Will the stock rise?") + resolved(40, 0.2, "Will the team win the league?"))
    assert table["other"]["earned"] and table["other"]["markets"] == 40 and table["other"]["t"] > 2.5
    assert not table["price"]["earned"] and table["price"]["t"] > 2.5  # one market short
    assert not table["sport"]["earned"] and table["sport"]["t"] < 0  # the price was closer
    held_no = {"side": "no", "bought": 0.4}  # a No bet gains when the Yes price falls
    assert "is now 70% (+75%)" in markets.guidance(held_no, 0.30, 0.3)[1] and markets.guidance(held_no, 0.30, 0.6)[0] == "hold"
    kind, advice = markets.guidance(held_no, 0.08, 0.5)  # nearly won: little left to gain
    assert kind == "cash out" and "is now 92% (+130%)" in advice and "at most +9%" in advice

    state, link = {}, "[A?](https://polymarket.com/market/a)"
    assert markets.price_move(link, 0.30, state) == [] and markets.price_move(link, 0.36, state) == []  # first sight, then a small move
    assert "from 30% to 46%" in markets.price_move(link, 0.46, state)[0] and markets.price_move(link, 0.47, state) == []

    # A whole run, then a second one after the market resolves.
    row = {
        "id": "polymarket-a", "question": "Will A happen?", "url": "https://polymarket.com/market/a",
        "resolution_criteria": "Resolves Yes if A happens.", "market_info_close_datetime": (NOW + timedelta(days=10)).isoformat(),
        "freeze_datetime": NOW.isoformat(), "price": 0.5,
    }
    private = hooks.STATE_DIR / "private"
    private.mkdir()
    (private / "watch.txt").write_text("https://polymarket.com/market/a yes bought=0.32 target=0.45\nhttps://polymarket.com/market/gone\n", encoding="utf-8")
    listed = {"https://polymarket.com/market/a": [row]}
    outcome = [None]
    with setting(markets, "PRIVATE", private), setting(tools, "polymarket", lambda url: listed[url]):
        with setting(tools, "busiest_markets", lambda count, soonest, latest: []), setting(tools, "polymarket_outcome", lambda slug: outcome[0]):
            with setting(hooks, "poll", lambda hook_list, days, until=None: [item("A is confirmed")]):
                questions, summary = markets.watch(make_bot(), tell=True)
                first_alert = "".join(path.read_text(encoding="utf-8") for path in (private / "alerts").iterdir())
                outcome[0] = 1
                time.sleep(1.1)  # alert files are named by the second
                markets.watch(make_bot(), tell=True)
    ledger = [json.loads(line) for line in (private / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [entry.get("forecast", entry.get("outcome")) for entry in ledger] == [0.7, 1] and ledger[0]["price"] == 0.5
    assert "- Forecast: [Will A happen?]" in first_alert and "Cash out: your Yes bet bought at 32% is now 50% (+56%), at your target of 45%" in first_alert
    # The bot said 70% against a price of 50%, so two rules practise on it with fake coins, on opposite sides.
    assert "Practice wallets (fake coins): 2 bought and 0 sold this run." in first_alert
    assert "Fake-coin wallets (each started with 1,000): Back the bot 999, Fade the bot 999, Back the favourite 1,000" in first_alert
    bets = [json.loads(line) for line in (private / "bets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(bet["rule"], bet["action"], bet["side"], bet["coins"]) for bet in bets] == [("Back the bot", "buy", "yes", 20.0), ("Fade the bot", "buy", "no", 20.0)]
    assert markets.UNTESTED in first_alert and "bet" not in "\n".join(summary)  # the public page never shows a bet
    alerts = sorted((private / "alerts").iterdir())
    assert len(alerts) == 2 and "Resolved Yes" in alerts[1].read_text(encoding="utf-8") and "bot right 100%" in alerts[1].read_text(encoding="utf-8")
    assert "Practice result (fake coins): Back the bot won 19, Fade the bot lost 20" in alerts[1].read_text(encoding="utf-8")
    assert "Back the bot 1,019, Fade the bot 980" in alerts[1].read_text(encoding="utf-8")
    card = (private / "scorecard.md").read_text(encoding="utf-8")
    assert "## Practice betting with fake coins" in card and "| Back the bot | 0 | 1 | 1 | +96% |" in card
    assert "**Referee:** 2 practice trades replayed, none broke a rule." in card and "Goal: turn 1,000 coins into 5,000" in card
    assert "### Latest practice trades" in card and "Back the bot bought 39 Yes at 51¢ for 20 coins (bot 70%, bid 49¢, ask 51¢, fee 0.00)" in card
    assert "Markets forecast: 1. Resolved: 1." in card and "| Bot | 100% |" in card and "Could not read the market at https://polymarket.com/market/gone" in card


def test_fake_coin_wallets_trade_at_live_prices_pay_fees_and_earn_a_say_only_by_results():
    at = lambda minutes: (NOW + timedelta(minutes=minutes)).isoformat()  # noqa: E731
    assert [rule(0.7, 0.5) for rule in wallet.RULES.values()] == ["yes", "no", None, None, "yes"]
    assert [rule(0.05, 0.12) for rule in wallet.RULES.values()] == [None, None, "no", "no", None]  # a favourite, and the bot is very sure
    # Back the bot keeps a position until its reason has gone; the price's-side rules hold to the end.
    assert wallet.back_the_bot(0.7, 0.65, held="yes") == "yes" and wallet.back_the_bot(0.7, 0.69, held="yes") is None
    assert wallet.back_the_bot(0.3, 0.5, held="yes") == "no"  # the bot changed its mind: switch sides
    assert wallet.back_the_favourite(0.5, 0.65, held="yes") == "yes" and wallet.fade_the_bot(0.3, 0.3, held="no") == "no"

    def market(n, forecast, bid, ask, days=3, **extra):
        return {
            "id": f"m{n}", "question": f"Q{n}?", "url": f"https://example.org/{n}", "forecast": forecast, "price": (bid + ask) / 2,
            "bid": bid, "ask": ask, "market_info_close_datetime": (NOW + timedelta(days=days)).isoformat(), **extra,
        }

    assert wallet.quotes({"price": 0.5}) == (0.49, 0.51) and wallet.quotes(market(1, 0.7, 0.48, 0.52)) == (0.48, 0.52)
    trades = wallet.trade([market(1, 0.7, 0.48, 0.52), market(2, 0.5, 0.49, 0.51)], [], {}, at(1))
    assert [(t["rule"], t["id"], t["action"], t["side"], t["price"], t["coins"]) for t in trades] == [
        ("Back the bot", "m1", "buy", "yes", 0.52, 20.0),  # a buyer pays the ask
        ("Fade the bot", "m1", "buy", "no", 0.52, 20.0),  # a No share costs 1 minus the Yes bid
    ]
    assert wallet.trade([market(1, 0.7, 0.48, 0.52)], trades, {}, at(2)) == []  # nothing changed: hold
    table = wallet.standings(trades, {}, {"m1": market(1, 0.7, 0.58, 0.62)})
    assert round(table["Back the bot"]["coins"], 2) == round(980 + 20 / 0.52 * 0.58, 2)  # open bets are worth what they sell for
    assert round(wallet.standings(trades, {}, {})["Back the bot"]["coins"], 2) == 1000.0  # no quote: valued at cost

    # The price comes to the bot: Back the bot cashes out at the bid. Fade the bot holds its side to the end.
    later = wallet.trade([market(1, 0.7, 0.68, 0.70)], trades, {}, at(3))
    assert [(t["rule"], t["action"], t["side"], round(t["price"], 2)) for t in later] == [("Back the bot", "sell", "yes", 0.68)]
    assert round(later[0]["coins"], 2) == round(20 / 0.52 * 0.68, 2) and wallet.book("Back the bot", trades + later, {})[2][0] > 0
    # The bot changes its mind instead: Back the bot sells its side and buys the other one.
    switched = wallet.trade([market(1, 0.3, 0.48, 0.52)], trades, {}, at(3))
    assert [(t["rule"], t["action"], t["side"]) for t in switched] == [("Back the bot", "sell", "yes"), ("Back the bot", "buy", "no")]

    # Polymarket's fee: shares x rate x price x (1 - price), paid buying and selling. It is heaviest on cheap shares.
    charged = market(8, 0.3, 0.09, 0.10, fee_rate=0.05, fee_exponent=1.0, min_shares=5.0)
    assert abs(wallet.fee(charged, 0.10) - 0.0045) < 1e-12 and wallet.fee(market(1, 0.7, 0.48, 0.52), 0.5) == 0.0
    bought = [t for t in wallet.trade([charged], [], {}, at(1)) if t["rule"] == "Back the bot"][0]
    assert bought["price"] == 0.10 and abs(bought["shares"] - 20 / 0.1045) < 0.001 and abs(bought["fee"] - bought["shares"] * 0.0045) < 0.001
    assert abs(bought["shares"] * 0.10 + bought["fee"] - 20.0) < 0.01  # the stake covers the shares and the fee
    sold = wallet.trade([{**charged, "bid": 0.29, "ask": 0.30, "price": 0.295}], [bought], {}, at(2))[0]
    assert sold["action"] == "sell" and abs(sold["coins"] - bought["shares"] * (0.29 - 0.05 * 0.29 * 0.71)) < 0.01
    assert wallet.audit([bought, sold], {}, {}) == []
    assert "did not pay the market's published fee" in wallet.audit([{**bought, "fee": 0.0, "shares": 200.0}], {}, {})[0]
    assert wallet.trade([{**charged, "min_shares": 500.0}], [], {}, at(1)) == []  # under the smallest order Polymarket takes

    # A wallet limited to one kind of market opens bets there and nowhere else. No wallet bets on a market months away.
    kinds = wallet.trade([market(6, 0.7, 0.48, 0.52, kind="price"), market(7, 0.7, 0.48, 0.52, kind="other")], [], {}, at(1))
    assert [t["id"] for t in kinds if t["rule"] == "Bot on price markets"] == ["m6"]
    assert [t["id"] for t in kinds if t["rule"] == "Back the bot"] == ["m6", "m7"]
    assert wallet.trade([market(9, 0.7, 0.48, 0.52, days=40)], [], {}, at(1)) == []
    assert wallet.trade([{**market(9, 0.7, 0.48, 0.52), "market_info_close_datetime": ""}], [], {}, at(1)) == []

    # Risk of loss: a position held to the end pays 1 a share or nothing. A thin or unquoted market is not traded.
    coins, held, gains = wallet.book("Back the bot", trades, {"m1": 0})
    assert coins == 980.0 and not held and gains == [-1.0]
    coins, _, gains = wallet.book("Back the bot", trades, {"m1": 1})
    assert round(coins, 2) == round(980 + 20 / 0.52, 2) and round(gains[0], 3) == round(1 / 0.52 - 1, 3)
    assert wallet.trade([market(3, 0.7, 0.48, 0.52, liquidity=100.0)], [], {}, at(1)) == []
    assert wallet.trade([market(4, 0.999, 0.985, 0.995)], [], {}, at(1)) == []  # nothing to gain above 99 cents
    empty_book = {**market(1, 0.3, 0.0, 1.0), "price": 0.6}
    assert wallet.trade([empty_book], trades, {}, at(2)) == []  # nobody to sell to: the position stays, even though the bot changed its mind
    assert round(wallet.positions(trades, {}, {"m1": empty_book})[0]["now"], 2) == 0.6  # valued at the last shown price

    # The referee replays the log: the real one is clean, and a doctored trade is caught.
    assert wallet.audit(trades + later, {}) == [] and all(t["rules"] == wallet.RULES_VERSION for t in trades)
    cheap = [{**trades[0], "price": 0.40, "shares": 50.0}]  # pretends to have bought 12 cents under the asking price
    assert "not bought at the live asking price" in wallet.audit(cheap, {})[0]
    greedy = [{**trades[0], "coins": 500.0, "shares": round(500 / 0.52, 4)}]  # a stake of half the wallet
    assert "more than 2% of its 1000.00 spare coins" in wallet.audit(greedy, {})[0]
    assert "after the market's result was known" in wallet.audit(trades, {"m1": 1}, {"m1": at(0)})[0]
    assert "sold a position it did not hold" in wallet.audit(later, {})[0]
    assert "made under rules version 2" in wallet.audit([{**trades[0], "rules": 2}], {})[0]
    # A wallet is paid when its market resolves, so later stakes are judged against the coins it then has.
    paid = wallet.book("Back the bot", trades, {"m1": 1})[0]  # 980 spare plus the payout
    after = [{**trades[0], "id": "m5", "at": at(5), "coins": round(0.02 * paid, 2), "shares": round(round(0.02 * paid, 2) / 0.52, 4)}]
    assert wallet.audit(trades[:1] + after, {"m1": 1}, {"m1": at(2)}) == []  # 2% of the larger wallet is within the limit
    assert "more than 2%" in wallet.audit(trades[:1] + after, {}, {})[0]  # without the win it would not be

    # Forty closed bets that mostly win earn a rule its say; thirty-nine do not, and neither does a losing run.
    def run(count, wins):
        bought = [
            {"rule": "Back the bot", "id": f"m{n}", "action": "buy", "side": "yes", "price": 0.5, "coins": 10.0, "shares": 20.0, "bid": 0.48, "ask": 0.5}
            for n in range(count)
        ]
        return bought, {f"m{n}": int(n % 10 < wins) for n in range(count)}

    assert wallet.standings(*run(40, 8), {})["Back the bot"]["earned"] and not wallet.standings(*run(39, 8), {})["Back the bot"]["earned"]
    assert not wallet.standings(*run(40, 4), {})["Back the bot"]["earned"]

    # The learner wallet buys nothing until a fixed rule has ten closed bets and is ahead; then it copies that rule.
    assert wallet.leader(trades, {}) is None and not [t for t in trades if t["rule"] == wallet.LEARNER]
    assert wallet.leader(*run(10, 8)) == "Back the bot" and wallet.leader(*run(9, 8)) is None and wallet.leader(*run(10, 4)) is None
    won_log, results = run(10, 8)
    copied = wallet.trade([market(99, 0.7, 0.48, 0.52)], won_log, results, at(9))
    assert [(t["rule"], t["side"]) for t in copied] == [("Back the bot", "yes"), ("Fade the bot", "no"), (wallet.LEARNER, "yes")]

    # The goal is reached only by winning: a wallet at 5,000 opens nothing new, and its course is reported honestly.
    rich = [{"rule": "Back the bot", "id": "big", "action": "buy", "side": "yes", "price": 0.01, "coins": 50.0, "shares": 5000.0, "bid": 0.0, "ask": 0.01}]
    assert not [t for t in wallet.trade([market(98, 0.7, 0.48, 0.52)], rich, {"big": 1}, at(1)) if t["rule"] == "Back the bot"]
    ahead, behind = wallet.standings(*run(10, 8), {})["Back the bot"], wallet.standings(*run(10, 4), {})["Back the bot"]
    assert wallet.course(ahead).startswith("about ") and wallet.course(behind) == "not on course"
    assert wallet.course(wallet.standings(trades, {}, {})["Back the bot"]) == "too early to say"

    rows = [{"forecast": 0.7, "crowd": 0.5, "outcome": 1}, {"forecast": 0.7, "crowd": 0.5, "outcome": 0}, {"forecast": 0.7, "crowd": 0.995, "outcome": 1}]
    backed = wallet.backtest(rows)["Back the bot"]
    assert (backed["bets"], backed["won"]) == (2, 1) and abs(backed["per_bet"] - (1 / 0.51 - 2) / 2) < 1e-9  # the settled price is skipped


def test_forecasters_start_from_the_market_price_only_when_asked_to():
    priced = make_question(community_prediction_at_access_time=0.84)
    assert "The market price" not in asyncio.run(make_bot().run_research(priced))
    with setting(autopilot, "ANCHOR", True):
        briefing = asyncio.run(make_bot().run_research(priced))
        assert briefing.startswith("## The market price") and "price Yes at 84%" in briefing
        assert "The market price" not in asyncio.run(make_bot().run_research(make_question()))  # no price, no note


def test_exam_can_keep_to_some_sources_and_to_questions_another_setup_sat():
    rows = [{"id": str(n), "source": source, "question": f"Q{n}?"} for n, source in enumerate(["polymarket", "metaculus", "kalshi", "polymarket"])]
    sat_by = {"new": [], "old": [{"id": "0"}, {"id": "1"}]}
    taken = []

    async def fake_sit_set(todo, label, set_date):
        taken.append([row["id"] for row in todo])

    with setting(bench, "exam", lambda set_date=None: rows), setting(bench, "results", lambda label: sat_by[label]):
        with setting(bench, "_sit_set", fake_sit_set):
            asyncio.run(bench.sit(10, "new", sources=("polymarket", "kalshi")))
            asyncio.run(bench.sit(10, "new", also_in="old"))
            asyncio.run(bench.sit(10, "new", sources=("polymarket",), also_in="old"))
    assert taken == [["0", "2", "3"], ["0", "1"], ["0"]]


def test_a_question_is_followed_more_closely_as_its_end_nears():
    soon, later, open_ended = (make_question(close_time=NOW + gap) if gap else make_question() for gap in (timedelta(hours=24), timedelta(days=60), None))
    assert autopilot.pace(soon, NOW) == (1.0, 4.0, 8.0)  # look hourly, forecast again after 4 hours, stale after 8
    assert autopilot.pace(later, NOW) == (12.0, 12.0, 168.0) and autopilot.pace(open_ended, NOW) == (3.0, 12, 168)
    last = NOW - timedelta(hours=5)
    held = dict(one_shot=False, already_forecasted=True, last_forecast_at=last, pending=["Rao resigns"], now=NOW)
    assert autopilot.decide(**held) is None and autopilot.decide(**held, cooldown_hours=4) == "news: Rao resigns"

    # A question looked at an hour ago is left alone when it closes in months, and looked at again when it closes tomorrow.
    polled = []
    for number, question in ((31, later), (32, soon)):
        question.id_of_question = number
        state = hooks.load_state(number)
        state.update(last_forecast_at=(NOW - timedelta(hours=1)).isoformat(), looked_at=(NOW - timedelta(hours=1)).isoformat())
        hooks.save_state(number, state)
    with setting(hooks, "poll", lambda hook_list, days, until=None: polled.append(1) or []):
        _, summary = asyncio.run(autopilot.run(make_bot(), [later, soon], one_shot=False))
    assert len(polled) >= 1 and any("Not due for a look yet: 1" in line for line in summary)
    assert any("No change" in line for line in summary)  # the one closing tomorrow was looked at


def test_series_facts_come_from_history_and_from_the_linked_symbol():
    start = datetime(2024, 1, 1).date()
    points = [(start + timedelta(days=n), 100 * 1.001**n) for n in range(1000)]  # grows 0.1% a day
    facts = tools.describe("Test index", points, horizon_days=30)
    assert "Latest value: 271.4" in facts and "12-month range" in facts and "over 7 days" in facts
    assert "+3.0% (median)" in facts and "past periods of 30 days" in facts  # 1.001 ** 30 is 3.0% up
    assert "past periods" not in tools.describe("Test index", points, horizon_days=0)
    assert tools.describe("Test index", points[:1], 30) == ""

    # A yearly series has no 7-day change, and a 46-day question sees at most one new value.
    yearly = [(datetime(2000 + n, 9, 30).date(), 1000.0 + 50 * n) for n in range(27)]
    facts = tools.describe("Annual outlays", yearly, horizon_days=46)
    assert "Previous value: 2,250, dated 2025-09-30" in facts and "about every 365 days" in facts
    assert "over 7 days" not in facts and "over 12 months" not in facts and "past periods" not in facts
    assert "At most 1 new value will" in facts
    monthly = [(datetime(2020 + n // 12, n % 12 + 1, 1).date(), 4.0 + 0.01 * n) for n in range(80)]
    facts = tools.describe("Unemployment", monthly, horizon_days=400)
    assert "over 12 months" in facts and "over 7 days" not in facts and "past periods of 400 days" in facts

    asked = []
    real_yahoo, real_fred = tools._yahoo, tools._fred
    tools._yahoo = lambda symbol: (asked.append(symbol), ("Yahoo series", points))[1]
    tools._fred = lambda series_id: (asked.append(series_id), ("FRED series", points))[1]
    try:
        assert "Yahoo series" in REAL_SERIES("https://finance.yahoo.com/quote/%5EGSPC/history?p=1", 30)
        assert "FRED series" in REAL_SERIES("https://fred.stlouisfed.org/series/unrate", 30)
        assert asked == ["^GSPC", "UNRATE"] and REAL_SERIES("https://example.org/page", 30) == ""
        tools._yahoo = lambda symbol: 1 / 0
        assert REAL_SERIES("https://finance.yahoo.com/quote/AAPL", 30) == ""  # a failed lookup is just skipped
    finally:
        tools._yahoo, tools._fred = real_yahoo, real_fred


def test_briefing_orders_facts_first_and_stays_under_the_comment_limit():
    question = make_question(resolution_criteria="Resolves per https://example.org/source.")
    evidence = [
        item("Rao wins vote", score=3) | {"url": "https://publisher.example/story"},
        item("Rao rally", score=2) | {"url": "https://news.google.com/rss/articles/CBMi"},
    ]
    real_page, real_series = tools.read_page, tools.series
    tools.read_page = lambda url, chars=tools.PAGE_CHARS: (f"TEXT OF {url} " * 3000)[:chars]
    tools.series = lambda url, horizon_days, until=None: "SERIES FACTS"
    try:
        text = asyncio.run(tools.briefing(question, evidence, hooks.digest(evidence), NOW))
        with setting(tools, "BRIEFING_LIMIT", 500):
            assert len(asyncio.run(tools.briefing(question, evidence, hooks.digest(evidence), NOW))) == 500
    finally:
        tools.read_page, tools.series = real_page, real_series
    order = [text.index(part) for part in ("Facts computed by code", "SERIES FACTS", "Pages the question names", "TEXT OF https://example.org/source")]
    assert order == sorted(order) and len(text) <= tools.BRIEFING_LIMIT
    assert "news.google.com" not in text  # only items with a real publisher link are read in full

    # The posted comment names the sources but does not republish their text.
    posted = tools.for_publication(text)
    assert "TEXT OF" not in posted and "### https://example.org/source" in posted
    assert "Facts computed by code" in posted and "SERIES FACTS" in posted and "Rao wins vote" in posted

    plain = asyncio.run(tools.briefing(make_question(), [], hooks.digest([]), NOW))
    assert "Headlines and posts" in plain and "Pages the question names" not in plain and "Full text" not in plain


def test_probe_reports_working_broken_and_picky_models():
    good = asyncio.run(bench.probe_model(named("a", "3 of 8 outcomes.\nProbability: 37.5%")))
    assert good["ok"] and good["readable"] and good["settings"] == "none sent"
    assert not asyncio.run(bench.probe_model(named("b", Http(410, "end of life"))))["ok"]
    vague = asyncio.run(bench.probe_model(named("c", "About a third.")))
    assert vague["ok"] and not vague["readable"] and "unreadable" in vague["note"]
    picky = named("d")
    picky.litellm_kwargs["extra_body"] = {"reasoning_effort": "high"}
    picky.reply = lambda prompt: Http(400) if "extra_body" in picky.litellm_kwargs else "Probability: 37.5%"
    assert asyncio.run(bench.probe_model(picky))["settings"] == "rejected ['extra_body']"


def test_replay_only_sees_news_from_before_the_replayed_day():
    until = datetime(2026, 10, 9, tzinfo=timezone.utc)
    asked = []

    def dated(day: datetime | None, title: str) -> str:
        when = f"<pubDate>{format_datetime(day)}</pubDate>" if day else ""
        return f"<item><title>{title} - Wire</title><link>x</link>{when}<source>Wire</source></item>"

    feed = "<rss><channel>" + dated(until - timedelta(days=4), "Before") + dated(until + timedelta(days=11), "After") + dated(None, "Undated") + "</channel></rss>"

    def fake_get(url, **params):
        asked.append((url, params))
        assert "bing" not in url, "Bing only has today's news and must not be asked about the past"
        if "google" in url:
            return FakeResponse(feed.encode())
        return FakeResponse(payload={"posts": [], "hits": []})

    real_get, hooks._get = hooks._get, fake_get
    try:
        found = hooks.poll([{"source": "news", "query": "Rao"}, {"source": "social", "query": "Rao"}], 14, until)
    finally:
        hooks._get = real_get
    assert [entry["title"] for entry in found] == ["Before"]  # nothing published later, nothing undated
    sent = {url.split("/")[2]: params for url, params in asked}
    assert sent["news.google.com"]["q"] == "Rao after:2026-09-25 before:2026-10-10"
    assert sent["api.bsky.app"]["until"] == "2026-10-09T00:00:00Z" and sent["api.bsky.app"]["since"] == "2026-09-25T00:00:00Z"
    assert f"created_at_i<{int(until.timestamp())}" in sent["hn.algolia.com"]["numericFilters"]


def test_replayed_clock_reaches_the_prompts_and_no_page_is_read():
    until = datetime(2026, 10, 9, tzinfo=timezone.utc)
    bot, question = make_bot(), make_question(resolution_criteria="Resolves per https://example.org/source.")
    bot.set_llm(FakeLlm("Probability: 30%"), "default")

    def forbidden(url, chars=0):
        raise AssertionError("a replay must not read live pages: they show today's content")

    cut_off = []
    real_page, real_series = tools.read_page, tools.series
    tools.read_page = forbidden
    tools.series = lambda url, horizon_days, until=None: cut_off.append(until) or ""
    try:
        with setting(autopilot, "AS_OF", until):
            assert autopilot._today() == "2026-10-09"
            asyncio.run(bot._binary_prompt_to_forecast(question, f"Today is {datetime.now():%Y-%m-%d}. Forecast."))
            text = asyncio.run(tools.briefing(question, [item("Rao wins vote", score=3)], "headlines", until, replay=True))
    finally:
        tools.read_page, tools.series = real_page, real_series
    assert bot.get_llm("default", "llm").prompts[0] == "Today is 2026-10-09. Forecast."
    assert "Today is 2026-10-09" in text and cut_off == [until.date()]  # data series stop at the replayed day
    assert autopilot._today() == f"{datetime.now(timezone.utc):%Y-%m-%d}"  # the real clock is back


def test_exam_keeps_resolved_market_questions_and_scores_them():
    def entry(source, key, question="Will it happen?", **extra):
        base = {
            "id": key, "source": source, "question": question, "url": "https://example.org/q", "background": "N/A",
            "resolution_criteria": "Resolves to the outcome of the linked question.",
            "market_info_resolution_criteria": "Resolves Yes if it happens.", "market_info_close_datetime": "2026-12-31T00:00:00+00:00",
            "freeze_datetime": "2026-10-09T00:00:00+00:00", "freeze_datetime_value": 0.2,
        }
        return {**base, **extra}

    asked = {"questions": [
        entry("metaculus", 42), entry("polymarket", "0xabc", freeze_datetime_value="N/A"), entry("kalshi", "KX-1"),
        entry("yfinance", "AAPL"), entry("metaculus", 43),
    ]}
    resolved = {"resolutions": [
        {"id": 42, "source": "metaculus", "resolved": True, "resolved_to": 1.0},
        {"id": "0xabc", "source": "polymarket", "resolved": True, "resolved_to": 0.0},
        {"id": "KX-1", "source": "kalshi", "resolved": False, "resolved_to": 0.4},  # still open: not ground truth
        {"id": "AAPL", "source": "yfinance", "resolved": True, "resolved_to": 1.0},  # a data question, not a market one
        {"id": 43, "source": "metaculus", "resolved": True, "resolved_to": 0.37},  # no clean yes or no
    ]}
    real_download = bench._download
    bench._download = lambda name, url: asked if "questions" in name else resolved
    try:
        rows = bench.exam()
    finally:
        bench._download = real_download
    assert sorted((row["id"], row["outcome"], row["crowd"]) for row in rows) == [("0xabc", 0, None), ("42", 1, 0.2)]

    question = autopilot.as_question(rows[0])
    assert question.resolution_criteria == "Resolves Yes if it happens." and question.background_info is None
    assert question.open_time == datetime(2026, 10, 9, tzinfo=timezone.utc)
    assert question.id_of_question == autopilot.as_question(rows[0]).id_of_question  # the same id on every run

    marked = [{"forecast": 0.9, "outcome": 1, "crowd": 0.8}] * 8 + [{"forecast": 0.7, "outcome": 0, "crowd": 0.2}] * 2
    result = bench.score(marked)
    assert result["questions"] == 10 and result["accuracy"] == 0.8 and abs(result["brier"] - (8 * 0.01 + 2 * 0.49) / 10) < 1e-9
    assert 0.45 < result["low"] < 0.55 and 0.94 < result["high"] < 0.97  # ten questions say very little
    assert bench.score(marked, "crowd")["accuracy"] == 1.0
    low, high = bench.wilson(90, 100)
    assert 0.82 < low < 0.84 and 0.94 < high < 0.95

    with setting(bench, "BENCH", hooks.STATE_DIR):
        lines = [json.dumps({**row, "question": f"Question {n}"}) for n, row in enumerate(marked)]
        (hooks.STATE_DIR / "results-trial.jsonl").write_text("\n".join(lines), encoding="utf-8")
        text = bench.report_text("trial")
    assert "| Bot | 80% |" in text and "| Crowd on the freeze date | 100% |" in text and "Largest misses" in text

    # Answering only when confident: forecasts inside the coin-flip band are left out.
    mixed = [{"forecast": chance, "outcome": int(chance > 0.5)} for chance in (0.1, 0.3, 0.5, 0.7, 0.75, 0.9)]
    assert [row["forecast"] for row in bench.confident(mixed)] == [0.1, 0.75, 0.9]
    # 0.9 is confident and 0.7 is not, so the 8 right answers count and the 2 wrong ones are skipped.
    assert "**100% right** on 8 of 10 questions (80% answered" in text

    # Tuning: stock questions are confident and half wrong, so a 90% rule has to leave them out.
    words = ["".join(pair) for pair in itertools.product("bdgjkmqz", repeat=2)]
    sat = [{"source": "kalshi", "question": f"Will the {word} stock rise?", "forecast": 0.9, "outcome": n % 2} for n, word in enumerate(words)]
    sat += [{"source": "kalshi", "question": f"Will {word} sign the bill?", "forecast": 0.9, "outcome": 1} for word in words]
    sat += [{"source": "kalshi", "question": f"Will {word} veto the bill?", "forecast": 0.72, "outcome": n % 2} for n, word in enumerate(words)]
    rule = bench.choose_rule(sat, 0.9)
    assert rule["above"] == 0.75 and rule["skip"] == ["kind price"] and len(bench.answered(sat, rule)) == len(words)
    assert bench.choose_rule([row for row in sat if "stock" in row["question"]], 0.9) is None  # nothing reaches it
    assert tools.family({"question": "Will oil be >98.99 on Nov 3? [Yes: $99 or above]"}) == tools.family({"question": "Will oil be >74.99 on Nov 3? [Yes: $75 or above]"})
    with setting(bench, "BENCH", hooks.STATE_DIR):
        (hooks.STATE_DIR / "results-tuned.jsonl").write_text("\n".join(json.dumps(row) for row in sat), encoding="utf-8")
        text = bench.tune_text("tuned", 0.9)
    assert text.count("leaving out kind price") == 3 and text.count("On the other half: 100% right") == 2


def test_run_skips_questions_about_to_close_and_gathers_evidence_in_parallel():
    busy, peak, lock = [0], [0], threading.Lock()

    def slow_poll(hook_list, days, until=None):
        with lock:
            busy[0] += 1
            peak[0] = max(peak[0], busy[0])
        time.sleep(0.3)
        with lock:
            busy[0] -= 1
        return [item("Rao sworn in as deputy PM")]

    closing = make_question(id_of_question=1, close_time=NOW + timedelta(minutes=5))
    later = [make_question(id_of_question=n, close_time=NOW + timedelta(hours=n)) for n in (2, 3, 4)]
    real_poll, hooks.poll = hooks.poll, slow_poll
    try:
        reports, summary = asyncio.run(autopilot.run(make_bot(), [later[2], closing, later[0], later[1]], one_shot=True))
    finally:
        hooks.poll = real_poll
    assert len(reports) == 3 and "closes in under 10 minutes" in summary[0]
    assert peak[0] >= 2  # evidence for several questions was gathered at the same time
    assert not (hooks.STATE_DIR / "1.json").exists()  # nothing was spent on the question that is closing


def test_watch_list_accepts_questions_and_tournaments_and_skips_bad_lines():
    class StubClient:
        def get_all_open_questions_from_tournament(self, slug):
            assert slug == "metaculus-cup-fall-2026"
            return [make_question(id_of_question=1), make_question(id_of_question=2)]

        def get_question_by_url(self, url, group_question_mode):
            if "404" in url:
                raise ValueError("not found")
            if "prize" in url:  # a bot-tournament question must never be re-forecast
                return make_question(id_of_question=9, tournament_slugs=["fall-futureeval-2026"])
            if "closed" in url:
                return make_question(id_of_question=3, state=QuestionState.CLOSED)
            return [make_question(id_of_question=2), make_question(id_of_question=4)]  # a group; 2 is a repeat

    (hooks.STATE_DIR / "watch_questions.txt").write_text(
        "# a comment\n\n"
        "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/  # the Cup\n"
        "https://www.metaculus.com/questions/404/gone/\n"
        "https://www.metaculus.com/questions/5/closed/\n"
        "https://www.metaculus.com/questions/7/prize/\n"
        "https://www.metaculus.com/questions/6/group/\n",
        encoding="utf-8",
    )
    with contextlib.chdir(hooks.STATE_DIR):
        found = autopilot.load_questions("watch", StubClient())
    assert sorted(question.id_of_question for question in found) == [1, 2, 4]


def test_scout_parses_fenced_json_and_reports_failure():
    question = make_question()
    found = asyncio.run(autopilot.scout(FakeLlm(SCOUT_REPLY), question, known="- [2026-11-12] Reuters: Rao named"))
    assert found == [
        {"source": "news", "query": "Asha Rao prime minister"},
        {"source": "news", "query": "site:thehindu.com coalition talks"},
        {"source": "social", "query": "Rao PM bid"},
    ]
    assert asyncio.run(autopilot.scout(FakeLlm("I cannot help with that."), question, "")) is None
    assert asyncio.run(autopilot.scout(FakeLlm(TimeoutError("slow")), question, "")) is None


def test_the_planner_can_name_a_price_series_but_only_a_ticker():
    reply = '{"news": ["oil price"], "social": [], "symbols": ["CL=F", "https://evil.example/x", "BTC-USD", "^GSPC"]}'
    found = asyncio.run(autopilot.scout(FakeLlm(reply), make_question(), ""))
    # Two at most, and anything that is not shaped like a ticker is dropped.
    assert found == [{"source": "news", "query": "oil price"}, {"source": "series", "query": "CL=F"}]
    assert hooks.poll([{"source": "series", "query": "CL=F"}], 14) == []  # a series hook is never polled for news

    start = datetime(2024, 1, 1).date()
    history = [(start + timedelta(days=n), 60 + n * 0.01) for n in range(1000)]
    asked = []
    real_yahoo, tools._yahoo = tools._yahoo, lambda symbol: (asked.append(symbol), ("Crude Oil (CL=F), from Yahoo Finance", history))[1]
    try:
        facts = tools.symbol_facts("CL=F", 30, until=start + timedelta(days=500))
        text = asyncio.run(tools.briefing(make_question(), [], "headlines", NOW, symbols=["CL=F"]))
        tools._yahoo = lambda symbol: 1 / 0
        assert tools.symbol_facts("CL=F", 30) == ""  # a failed lookup is just skipped
    finally:
        tools._yahoo = real_yahoo
    assert asked == ["CL=F", "CL=F"] and "Latest value: 65.00" in facts  # cut off at the replayed day
    assert "picked by the search planner" in facts and "Crude Oil (CL=F)" in text


def test_llm_triage_takes_highest_rating_and_degrades_to_background():
    question, items = make_question(), [item("a"), item("b"), item("c")]
    scores = asyncio.run(autopilot.triage(FakeLlm('{"3": [2], "2": [1, 2], "1": [99]}'), question, items))
    assert scores == [2.0, 3.0, 0.0]
    assert asyncio.run(autopilot.triage(FakeLlm("no json here {"), question, items)) == [1.0, 1.0, 1.0]
    assert asyncio.run(autopilot.triage(FakeLlm("unused"), question, [])) == []


def test_jev_triage_and_fast_read_follow_the_documented_response_shape():
    question, items = make_question(), [item("a"), item("b")]
    sent = []

    def fake_post(url, headers, json, timeout):
        sent.append(json)
        answers = {
            name: {"type": "noul", "noul": 0.34} if spec["type"] == "noul"
            else {"type": "score", "score": 2.6 if name == "item0" else 0.2, "confidence": 0.5}
            for name, spec in json["questions"].items()
        }
        return FakeResponse(payload={"model": "jev-1.13.0", "answers": answers})

    real_post, autopilot.requests.post = autopilot.requests.post, fake_post
    os.environ["TYPESAFE_API_KEY"] = "test-key"
    try:
        llm = FakeLlm(AssertionError("the LLM must not be asked when Jev answers"))
        assert asyncio.run(autopilot.triage(llm, question, items)) == [2.6, 0.2]
        assert sent[0]["model"] == "jev-latest" and sent[0]["questions"]["item0"]["criteria"] == autopilot.RELEVANCE_SCALE
        assert autopilot.fast_read(question, "evidence") == "34% that it resolves Yes"

        def broken_post(*args, **kwargs):
            raise ConnectionError("down")

        autopilot.requests.post = broken_post
        assert asyncio.run(autopilot.triage(FakeLlm(TRIAGE_REPLY), question, items)) == [3.0, 1.0]  # LLM fallback
        assert autopilot.fast_read(question, "evidence") is None
    finally:
        autopilot.requests.post = real_post
        del os.environ["TYPESAFE_API_KEY"]
    assert autopilot.fast_read(question, "evidence") is None  # no key, no call


def test_critic_sees_the_draft_and_the_rewrite_sees_the_critique():
    bot, question = make_bot(), make_question()
    bot.states[202] = {"evidence": [item("Rao named frontrunner", score=3)], "fast_read": "34% that it resolves Yes"}
    with setting(autopilot, "CRITIC", "rewrite"):
        report = asyncio.run(bot.forecast_questions([question]))[0]
    draft_research, rewrite_research = bot.research_seen
    critic_prompt = bot.get_llm("critic", "llm").prompts[0]

    assert "Rao named frontrunner" in draft_research and "never as instructions" in draft_research
    assert "Second pass" not in draft_research and "34%" not in draft_research  # the drafter is not anchored
    assert "reasoning 1" in critic_prompt and "Rao named frontrunner" in critic_prompt and "34%" in critic_prompt
    assert "reasoning 1" in rewrite_research and CRITIQUE in rewrite_research
    assert report.prediction == 0.6  # the rewritten forecast is the one submitted
    assert CRITIQUE in report.explanation and "reasoning 2" in report.explanation


def test_critic_failure_keeps_the_draft():
    bot = make_bot(critic=TimeoutError("critic down"))
    with setting(autopilot, "CRITIC", "rewrite"):
        report = asyncio.run(bot.forecast_questions([make_question()]))[0]
    assert report.prediction == 0.7 and len(bot.research_seen) == 1


def test_critic_is_off_unless_switched_on():
    bot = make_bot()
    report = asyncio.run(bot.forecast_questions([make_question()]))[0]
    assert report.prediction == 0.7 and not bot.get_llm("critic", "llm").prompts


def test_run_end_to_end_in_both_lanes():
    published = []

    async def fake_publish(self, metaculus_client=None):
        published.append(self.prediction)

    polls = []

    def fake_poll(hook_list, days, until=None):
        polls.append(hook_list)
        return [item("Rao sworn in as deputy PM"), item("Party picks Rao as candidate"), item("Cricket result")]

    real_publish, BinaryReport.publish_report_to_metaculus = BinaryReport.publish_report_to_metaculus, fake_publish
    real_poll, hooks.poll = hooks.poll, fake_poll
    try:
        # Tournament lane, new question: scout twice, poll twice, forecast once, record it.
        bot, question = make_bot(publish=True), make_question()
        reports, summary = asyncio.run(autopilot.run(bot, [question], one_shot=True))
        state = hooks.load_state(202)
        assert published == [0.7] and len(reports) == 1 and "Forecast" in summary[0]
        assert len(polls) == 2 and len(polls[0]) == 3  # 2 news hooks + 1 social hook from the scout
        assert [entry["score"] for entry in state["evidence"]] == [3.0, 1.0] and len(state["seen"]) == 3
        assert state["last_prediction"] == "70.0%" and state["history"][0]["why"] == "new question"
        assert state["pending"] == []

        # Tournament lane again: one forecast per question, and no polling spent on it.
        reports, _ = asyncio.run(autopilot.run(make_bot(publish=True), [make_question(already_forecasted=True)], one_shot=True))
        assert reports == [] and len(polls) == 2 and published == [0.7]

        # Watch lane, 13 hours later, a decisive new headline arrives: the forecast is updated.
        long_ago = (NOW - timedelta(hours=13)).isoformat()
        state.update(last_forecast_at=long_ago, looked_at=long_ago)
        hooks.save_state(202, state)
        hooks.poll = lambda hook_list, days, until=None: [item("Rao wins confidence vote")]
        watched = make_question(already_forecasted=True)
        reports, summary = asyncio.run(autopilot.run(make_bot(publish=True), [watched], one_shot=False))
        state = hooks.load_state(202)
        assert published == [0.7, 0.7] and state["history"][-1]["why"] == "news: Rao wins confidence vote"

        # Watch lane, right after that update: the question is not due for another look yet.
        hooks.poll = lambda hook_list, days, until=None: [item("President invites Rao to form government")]
        reports, summary = asyncio.run(autopilot.run(make_bot(publish=True), [watched], one_shot=False))
        assert reports == [] and "Not due for a look yet: 1" in summary[0] and hooks.load_state(202)["pending"] == []

        # Four hours on it is looked at again, and more news has come: held back by the cooldown but not lost.
        state = hooks.load_state(202)
        state["looked_at"] = (NOW - timedelta(hours=4)).isoformat()
        hooks.save_state(202, state)
        reports, summary = asyncio.run(autopilot.run(make_bot(publish=True), [watched], one_shot=False))
        assert reports == [] and "No change" in summary[0]
        assert hooks.load_state(202)["pending"] == ["President invites Rao to form government"]

        # A dry run forecasts but must not be recorded as the real forecast.
        hooks.poll = fake_poll
        fresh = make_question(id_of_question=303)
        asyncio.run(autopilot.run(make_bot(publish=False), [fresh], one_shot=True))
        assert hooks.load_state(303)["last_forecast_at"] is None and len(published) == 2
    finally:
        BinaryReport.publish_report_to_metaculus = real_publish
        hooks.poll = real_poll


if __name__ == "__main__":
    tests = [check for name, check in sorted(globals().items()) if name.startswith("test_")]
    for check in tests:
        with tempfile.TemporaryDirectory() as folder:
            hooks.STATE_DIR = Path(folder)
            check()
        print(f"ok  {check.__name__}")
    print(f"\n{len(tests)} checks passed")
