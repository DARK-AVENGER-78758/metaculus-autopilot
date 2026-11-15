"""The model pool: free NVIDIA NIM models behind one rate limit, with automatic fallback.

A Pool looks like a single model to the rest of the bot. Each call goes to the next healthy
member; a member that fails hands the call to the next one, and a member NVIDIA has retired
is skipped for the rest of the run. Any litellm model name works as a member, so another
provider can be mixed in through FORECAST_MODELS or UTILITY_MODELS without code changes.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
import time

import litellm
from forecasting_tools import GeneralLlm, RefreshingBucketRateLimiter

logger = logging.getLogger(__name__)


def _names(variable: str, default: str) -> list[str]:
    return [name.strip() for name in (os.getenv(variable) or default).split(",") if name.strip()]


# One per model family, fastest first, chosen by timing a real 16,000-character forecasting
# prompt on 2026-11-15 with all models running at once: Nemotron 3 Ultra 39 s, MiniMax M3.1
# 220 s, StepFun Step-5 291 s, Gemma 4 382 s. Forecasts come from the first WIDTH healthy
# models; Gemma is the spare. Left out: GLM 5.3, Mistral Large 4 and DeepSeek V4.1 Flash
# never finished a real prompt, and Kimi K2.6 and Nemotron 3 Super were not available to the
# account. A short test question is no guide: Mistral answered one in 4 seconds.
FORECAST_MODELS = _names(
    "FORECAST_MODELS",
    "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b,nvidia_nim/minimaxai/minimax-m3.1-flash,"
    "nvidia_nim/stepfun-ai/step-5-preview,nvidia_nim/google/gemma-4-31b-it",
)
# Search plans, headline scoring and answer repair: the first healthy one does the work.
# These are bulk jobs, so they run without reasoning (see REASONING below). Scoring 120
# headlines took Nemotron 3 Ultra 8 s, MiniMax 47 s and Gemma 109 s.
UTILITY_MODELS = _names(
    "UTILITY_MODELS",
    "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b,nvidia_nim/minimaxai/minimax-m3.1-flash,"
    "nvidia_nim/google/gemma-4-31b-it",
)
WIDTH = 3  # model families that share the forecasting; the rest are spares

# NVIDIA allows about 40 requests a minute per key, and the tournament and watch jobs can run
# at the same time, so each process takes under half. Raise it once `bench.py probe` has
# measured the real limit.
REQUESTS_PER_MINUTE = float(os.getenv("REQUESTS_PER_MINUTE") or 18)
RATE_LIMIT_WAIT = 15  # seconds to wait after a 429; NVIDIA sends no Retry-After header
TIMEOUT = 300  # seconds of silence allowed on a connection; NVIDIA drops it at about this point
CALL_LIMIT = 600  # seconds one streamed call may take in total
STRIKES = 2  # timeouts after which a model is benched for the rest of the run

# Reasoning settings per model: (for forecasting, for utility work). Forecasts get the most
# reasoning the model offers; utility jobs get none, which is twice as fast. The library
# drops `reasoning_effort` unless it travels in `extra_body`. If a model rejects its settings
# the pool retries without them, so a wrong entry costs one call, not a forecast.
REASONING = {
    "nemotron-3-ultra": ({"reasoning_effort": "high"}, {"reasoning_effort": "none"}),
}
OPTIONAL_SETTINGS = ("extra_body", "max_tokens")


class Member(GeneralLlm):
    """One model in a pool, read as a stream."""

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        # Streamed, because NVIDIA closes a connection that stays silent for 300 seconds and
        # the slower free models need longer than that for a full forecast. Only the answer
        # is kept. The model's private reasoning arrives in a separate field and is dropped,
        # so a number from its scratch work can never be read as the forecast.
        litellm.drop_params = True
        stream = await litellm.acompletion(
            messages=self.model_input_to_message(prompt, system_prompt), stream=True, **self.litellm_kwargs
        )
        answer = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                answer.append(chunk.choices[0].delta.content)
        return "".join(answer)


def member(name: str, forecasting: bool = True) -> Member:
    # max_tokens leaves room for reasoning plus the answer. num_retries=0 stops litellm from
    # retrying on its own, which turned one 300-second timeout into a 15-minute wait.
    settings: dict = {"max_tokens": 16000 if forecasting else 6000, "num_retries": 0}
    reasoning = next((value for key, value in REASONING.items() if key in name), None)
    if reasoning:
        settings["extra_body"] = dict(reasoning[0 if forecasting else 1])
    # allowed_tries=1: the pool decides what to retry. The library's own retry would hammer a
    # retired model and bypass the rate limit.
    return Member(model=name, timeout=TIMEOUT, allowed_tries=1, **settings)


_limiter_for: tuple[asyncio.AbstractEventLoop, RefreshingBucketRateLimiter] | None = None


def _limiter() -> RefreshingBucketRateLimiter:
    """One limiter per event loop, shared by every pool, because the limit belongs to the API key."""
    global _limiter_for
    loop = asyncio.get_running_loop()
    if _limiter_for is None or _limiter_for[0] is not loop:
        # A small burst, then a steady refill. A bucket the size of the whole minute would
        # send the whole minute's calls at once.
        _limiter_for = (loop, RefreshingBucketRateLimiter(capacity=5, refresh_rate=REQUESTS_PER_MINUTE / 60))
    return _limiter_for[1]


def _status(error: Exception) -> int | None:
    return getattr(error, "status_code", None)


def strip_thinking(text: str) -> str:
    """Remove a reasoning model's <think> block, including one that was cut off before it closed."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"<think>.*", "", text, flags=re.S).strip()


class Pool(GeneralLlm):
    """Several models behind one name."""

    def __init__(self, role: str, members: list[GeneralLlm], width: int = 1) -> None:
        super().__init__(model=f"pool/{role}")
        self.role, self.members, self.width = role, members, width
        self.dead: dict[str, str] = {}  # model name -> why it was retired for this run
        self.calls: list[tuple[str, str, float]] = []  # (model, outcome, seconds)
        self.timeouts: dict[str, int] = {}  # model name -> timeouts so far
        self._turn = itertools.count()

    def healthy(self) -> list[GeneralLlm]:
        return [model for model in self.members if model.model not in self.dead]

    async def ask(self, prompt: str, system_prompt: str | None = None) -> tuple[str, str]:
        """Return (answer, name of the model that gave it)."""
        healthy = self.healthy()
        if not healthy:
            raise RuntimeError(f"Every {self.role} model is unavailable: {self.dead}")
        # Take turns among the first `width` healthy models; everything after is a spare.
        # If the chosen model fails, the others are tried in roster order, fastest first.
        chosen = healthy[next(self._turn) % min(self.width, len(healthy))]
        last_error: Exception | None = None
        for model in [chosen] + [other for other in healthy if other is not chosen]:
            try:
                return await self._call(model, prompt, system_prompt), model.model
            except Exception as error:
                last_error = error
                reason = f"{type(error).__name__}: {error}"[:300]  # a timeout has no message of its own
                if _status(error) in (404, 410) or "end of life" in str(error).lower():
                    self.dead[model.model] = reason
                    logger.error(f"{model.model} is gone, skipping it for this run: {reason}")
                    continue
                if "timeout" in type(error).__name__.lower():
                    self.timeouts[model.model] = self.timeouts.get(model.model, 0) + 1
                    if self.timeouts[model.model] >= STRIKES:
                        self.dead[model.model] = f"timed out {STRIKES} times in this run"
                logger.warning(f"{model.model} failed, trying the next model: {reason}")
        raise RuntimeError(f"No {self.role} model answered. Last error: {last_error}")

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        return (await self.ask(prompt, system_prompt))[0]

    async def _call(self, model: GeneralLlm, prompt: str, system_prompt: str | None) -> str:
        for attempt in range(3):
            await _limiter().wait_till_able_to_acquire_resources(1)
            started = time.monotonic()
            try:
                # A hard limit of our own: a call has been seen to outlive the library's timeout.
                answer = strip_thinking(await asyncio.wait_for(model.invoke(prompt, system_prompt), CALL_LIMIT))
                if not answer:
                    raise ValueError("empty answer")
            except Exception as error:
                self.calls.append((model.model, type(error).__name__, time.monotonic() - started))
                if _status(error) == 429 and attempt < 2:
                    await asyncio.sleep(RATE_LIMIT_WAIT)
                    continue
                if _status(error) in (400, 422) and any(
                    model.litellm_kwargs.pop(setting, None) is not None for setting in OPTIONAL_SETTINGS
                ):
                    logger.warning(f"{model.model} rejected its optional settings; retrying without them")
                    continue
                raise
            self.calls.append((model.model, "ok", time.monotonic() - started))
            return answer
        raise RuntimeError(f"{model.model} is still rate limited")  # unreachable: the last attempt re-raises

    def report(self) -> list[str]:
        """One summary line per model that was called or retired, for the run page."""
        lines = []
        for name in dict.fromkeys([model for model, _, _ in self.calls] + list(self.dead)):
            mine = [(outcome, seconds) for model, outcome, seconds in self.calls if model == name]
            good = [seconds for outcome, seconds in mine if outcome == "ok"]
            average = f", {sum(good) / len(good):.0f} s average" if good else ""
            retired = f" RETIRED: {self.dead[name]}" if name in self.dead else ""
            lines.append(f"- {self.role} `{name}`: {len(good)} answers, {len(mine) - len(good)} failures{average}{retired}")
        return lines

    def to_dict(self) -> dict:
        return {"pool": self.role, "members": [model.model for model in self.members], "width": self.width}


def pools() -> tuple[Pool, Pool]:
    """The forecasting pool and the utility pool, built from the environment."""
    return (
        Pool("forecast", [member(name) for name in FORECAST_MODELS], width=WIDTH),
        Pool("utility", [member(name, forecasting=False) for name in UTILITY_MODELS]),
    )
