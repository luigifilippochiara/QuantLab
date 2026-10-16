"""Narrator: a short plain-English summary of the results, with every number checked.

Role in the architecture: this is the second (and last) module that talks to
an LLM. The narrator never sees data series. It gets a FACTS SHEET: the key
numbers the executor computed, each already rounded and formatted. It is told
to use only those numbers, exactly as displayed, and to do no arithmetic.

Then `check_grounding` verifies that claim deterministically: it extracts every
number from the narrative and checks that each one is a correct rounding of a
computed fact (or its x100 percentage form), a plan parameter, a number from the
request or a warning, or an annualisation factor. Dates are facts too: every
date in the narrative must be a date of the facts sheet.
The UI shows "all numbers verified" or lists the ungrounded numbers. The
deterministic results table is always shown as the source of truth.

Without an API key (mock mode), or if the LLM call fails, a deterministic
template narrator writes the summary instead.
"""

import math
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from quantlab.capabilities import count_episodes
from quantlab.config import Settings
from quantlab.plan import Plan, describe_plan
from quantlab.planner import (
    FALLBACK_BETA,
    api_error_message,
    make_anthropic_client,
    make_openai_client,
)
from quantlab.types import ANNUALISATION_FACTORS, Artifact, DataType
from quantlab.validation import Issue

NARRATOR_PROMPT_VERSION = "narrator-v3"

SYSTEM_PROMPT = """\
You write short, plain-English summaries of quantitative analysis results for a bank's \
markets and risk teams.

Rules:
1. Use ONLY numbers that appear in the FACTS SHEET, written exactly as displayed. You may \
write a decimal as a percentage (0.1275 as 12.75%).
2. Do no arithmetic: no new differences, ratios, sums or rounding of your own.
3. Do not add numbers from your own knowledge (no dates, levels or thresholds that are not given).
4. Dates are facts too: mention only dates from the FACTS SHEET, written exactly as displayed \
(YYYY-MM-DD).
5. Mention every warning and approved assumption that affects how the results should be read.
6. Describe what was computed; do not claim causes, forecasts or statistical significance.
7. Write 3 to 6 sentences as one paragraph. No headings, no bullet points.
"""

# A number in text: digits with optional thousands separators and decimals.
# The look-behind skips digits glued to letters or dots, such as step ids (s7),
# guardrail codes (R04), dataset ids (DGS10) and release names (H.15).
NUMBER_PATTERN = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Dates written in words, so "on 19 November 2021" cannot slip past the date
# check as the plain numbers 19 and 2021. Full month names or their usual
# abbreviations; the \b keeps words such as "market" or "decline" from matching.
MONTH = (
    r"(january|february|march|april|may|june|july|august|september|october|november|"
    r"december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\b\.?"
)
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}  # fmt: skip
DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{MONTH}\s*,?\s+(\d{{4}})\b", re.I)
MONTH_DAY_YEAR = re.compile(rf"\b{MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(\d{{4}})\b", re.I)
MONTH_YEAR = re.compile(rf"\b{MONTH}\s+(\d{{4}})\b", re.I)


class Fact(BaseModel):
    """One computed number (or date) the narrator may use.

    Fields:
        fact_id: stable id, e.g. "s3.mean".
        label: what it is, e.g. "s3 summary_stats mean".
        value: the exact computed value.
        display: the rounded text the narrator must use, e.g. "0.0005631".
    """

    fact_id: str
    label: str
    value: float | int | str
    display: str


class GroundingResult(BaseModel):
    """Outcome of the numeric grounding check.

    Fields:
        grounded: True if every number in the narrative is accounted for.
        numbers_checked: how many numbers (and dates) were found.
        ungrounded: the numbers that could not be matched, as written.
    """

    grounded: bool
    numbers_checked: int
    ungrounded: list[str] = Field(default_factory=list)


class NarrativeResult(BaseModel):
    """A narrative and how it was produced.

    Fields:
        text: the narrative.
        source: "llm" or "template".
        model: model that wrote it (None for the template).
        prompt_version: narrator prompt version.
        latency_ms: time spent writing it.
        input_tokens: prompt tokens reported by the API.
        output_tokens: output tokens reported by the API.
        error: why the LLM was not used, if it failed and the template took over.
        facts: the facts sheet it was given.
        grounding: the result of the numeric grounding check (None until checked).
    """

    text: str
    source: str
    model: str | None = None
    prompt_version: str = NARRATOR_PROMPT_VERSION
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    facts: list[Fact]
    grounding: GroundingResult | None = None


# ---------------------------------------------------------------------------
# Facts sheet
# ---------------------------------------------------------------------------


def format_number(value: Any) -> str:
    """Format a value for display: integers with separators, floats to 4 significant figures.

    Args:
        value: the value.

    Returns:
        Display text. Never uses scientific notation, so the narrative can quote it.
    """
    if isinstance(value, (bool, str)):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0"
    # 4 significant figures: the number of decimals depends on the magnitude
    # (0.001234 -> 6 decimals; 27204.38 -> 0 decimals, all integer digits kept).
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(3 - magnitude, 0)
    return f"{value:,.{decimals}f}"


def build_facts_sheet(plan: Plan, artifacts: dict[str, Artifact]) -> list[Fact]:
    """Collect the key computed numbers of a run, rounded consistently.

    Series: observations, first/last date, latest value. Flags: counts, share
    flagged, episodes. Tables: every statistic.

    Args:
        plan: the executed plan (for step order and capability names).
        artifacts: step id -> output artifact.

    Returns:
        The facts, in plan order.
    """
    facts: list[Fact] = []

    def add(step_id: str, capability: str, key: str, value: Any) -> None:
        label = f"{step_id} {capability} {key.replace('_', ' ')}"
        display = format_number(value)
        facts.append(Fact(fact_id=f"{step_id}.{key}", label=label, value=value, display=display))

    for step in plan.steps:
        artifact = artifacts.get(step.id)
        if artifact is None:
            continue
        cap, meta = step.capability, artifact.meta
        if artifact.data_type == DataType.SERIES:
            add(step.id, cap, "n_obs", meta.n_obs)
            add(step.id, cap, "start", meta.start)
            add(step.id, cap, "end", meta.end)
            add(step.id, cap, "last_value", float(artifact.value.iloc[-1]))
            if meta.period_start is not None:  # filter_period: the resolved interval
                add(step.id, cap, "period_start", meta.period_start)
                add(step.id, cap, "period_end", meta.period_end)
        elif artifact.data_type == DataType.FLAGS:
            flags = artifact.value
            add(step.id, cap, "n_obs", meta.n_obs)
            add(step.id, cap, "n_flagged", int(flags.sum()))
            add(step.id, cap, "share_flagged", float(flags.mean()))
            add(step.id, cap, "n_episodes", count_episodes(flags))
        else:  # TABLE: named scalars, or named rows such as find_extremes' "max"
            for key, value in artifact.value.items():
                if isinstance(value, dict):  # e.g. "max_date", "max_value", "max_tie_count"
                    for column, cell in value.items():
                        add(step.id, cap, f"{key}_{column}", cell)
                else:
                    add(step.id, cap, key, value)
    return facts


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------


def _numbers_in(text: str) -> list[tuple[str, float, int]]:
    """Find numbers in text. Returns (as written, value, number of decimals) per number."""
    found = []
    for match in NUMBER_PATTERN.finditer(text):
        integer_part, fraction = match.group(1), match.group(2) or ""
        value = float(integer_part.replace(",", "") + fraction)
        decimals = len(fraction) - 1 if fraction else 0
        found.append((match.group(0), value, decimals))
    return found


def _written_dates(text: str) -> tuple[list[tuple[str, str | None]], str]:
    """Find dates written in words, e.g. "19 November 2021" or "Nov 19, 2021".

    Args:
        text: narrative text (ISO dates already removed).

    Returns:
        ([(as written, ISO date or None if no such day exists)], the text with
        those dates blanked out so their numbers are not checked again).
    """
    found: list[tuple[str, str | None]] = []

    def to_iso(match: re.Match, day: str, month: str, year: str) -> str:
        try:
            iso = date(int(year), MONTHS[month[:3].lower()], int(day)).isoformat()
        except ValueError:  # e.g. "31 February 2021": not a real day, so never grounded
            iso = None
        found.append((match.group(0), iso))
        return " "

    text = DAY_MONTH_YEAR.sub(lambda m: to_iso(m, m.group(1), m.group(2), m.group(3)), text)
    text = MONTH_DAY_YEAR.sub(lambda m: to_iso(m, m.group(2), m.group(1), m.group(3)), text)
    return found, text


def check_grounding(
    narrative: str, facts: list[Fact], plan: Plan, extra_texts: list[str] | None = None
) -> GroundingResult:
    """Check that every number and date in a narrative comes from the results or the inputs.

    A number written with d decimals is grounded if it is within half a unit of
    its last digit of an allowed value, i.e. it is a correct rounding of it.
    Allowed values: each fact's value and its x100 form (percentages), numbers
    in the facts' display strings, numeric plan parameters (and x100), numbers in
    `extra_texts` (the request, warning messages, dataset titles) and the
    annualisation factors. Dates are facts: a date (YYYY-MM-DD, or written in
    words such as "19 November 2021") is grounded only if it is a date of the
    facts sheet, and a month ("November 2021") only if a fact date falls in it.
    A year on its own is grounded if it lies within the date range of the facts.

    Limitation: this proves each number is a correct rounding of SOME allowed
    value, not that it is attached to the right concept in the sentence.

    Args:
        narrative: the text to check.
        facts: the facts sheet.
        plan: the plan (for parameter values).
        extra_texts: other trusted texts whose numbers are allowed.

    Returns:
        A GroundingResult.
    """
    allowed: list[float] = [float(factor) for factor in ANNUALISATION_FACTORS.values()]
    dates: list[str] = []
    for fact in facts:
        if isinstance(fact.value, str):
            if DATE_PATTERN.fullmatch(fact.value):
                dates.append(fact.value)
            continue
        if isinstance(fact.value, (int, float)) and math.isfinite(fact.value):
            allowed += [abs(fact.value), abs(fact.value) * 100]
        allowed += [value for _, value, _ in _numbers_in(fact.display)]
    for step in plan.steps:
        for value in step.params.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                allowed += [abs(value), abs(value) * 100]
    for text in extra_texts or []:
        allowed += [value for _, value, _ in _numbers_in(text)]

    first_date, last_date = (min(dates), max(dates)) if dates else ("", "")
    ungrounded: list[str] = []
    checked = 0

    # Dates first, removing them so "2026-10-09" is not read as 2026, 10 and 09.
    # A date must BE a fact: being inside the data range is not enough, or an
    # invented peak date between the first and last observation would pass.
    for iso in DATE_PATTERN.findall(narrative):
        checked += 1
        if iso not in dates:
            ungrounded.append(iso)
    written, remaining = _written_dates(DATE_PATTERN.sub(" ", narrative))
    for text, iso in written:
        checked += 1
        if iso not in dates:
            ungrounded.append(text)
    for match in MONTH_YEAR.finditer(remaining):
        checked += 1
        month = f"{match.group(2)}-{MONTHS[match.group(1)[:3].lower()]:02d}"
        if not any(d.startswith(month) for d in dates):
            ungrounded.append(match.group(0))
    remaining = MONTH_YEAR.sub(" ", remaining)

    for written, value, decimals in _numbers_in(remaining):
        checked += 1
        is_year = decimals == 0 and "," not in written and 1900 <= value <= 2100
        if is_year and dates and int(first_date[:4]) <= value <= int(last_date[:4]):
            continue
        # Half a unit of the last written digit, plus a hair for float error.
        tolerance = 0.5 * 10 ** (-decimals) + 1e-12
        if not any(abs(value - candidate) <= tolerance for candidate in allowed):
            ungrounded.append(written)
    return GroundingResult(grounded=not ungrounded, numbers_checked=checked, ungrounded=ungrounded)


# ---------------------------------------------------------------------------
# Template narrator (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _input_name(step: Any, artifacts: dict[str, Artifact]) -> str:
    """Name the series a step summarised, e.g. "the returns series (s2)"."""
    source = step.inputs.get("series", "")
    kind = artifacts[source].meta.kind if source in artifacts else None
    return f"the {kind} series ({source})" if kind else f"the series ({source})"


def template_narrative(
    plan: Plan, artifacts: dict[str, Artifact], facts: list[Fact], issues: list[Issue]
) -> str:
    """Write a deterministic summary from the facts sheet (used in mock mode and as fallback).

    Args:
        plan: the executed plan.
        artifacts: step id -> output artifact.
        facts: the facts sheet (all numbers are taken from its display strings).
        issues: warnings and approved assumptions to mention.

    Returns:
        The narrative text.
    """
    shown = {fact.fact_id: fact.display for fact in facts}
    sentences = []

    loads = [s for s in plan.steps if s.capability == "load_series" and s.id in artifacts]
    if loads:
        described = ", ".join(
            f"{s.params['dataset_id']} ({shown[s.id + '.start']} to {shown[s.id + '.end']})"
            for s in loads
        )
        sentences.append(f"The analysis used {described}.")

    for step in plan.steps:
        if step.id not in artifacts:
            continue
        sid, cap = step.id, step.capability
        # d: this step's facts by short name, e.g. d["mean"] == "0.0005631".
        d = {
            fact.fact_id.removeprefix(f"{sid}."): fact.display
            for fact in facts
            if fact.fact_id.startswith(f"{sid}.")
        }
        if cap in ("rolling_volatility", "spread", "difference"):
            what = {"rolling_volatility": "rolling volatility", "spread": "spread",
                    "difference": "change"}[cap]  # fmt: skip
            sentences.append(f"The latest {what} ({sid}) is {d['last_value']} on {d['end']}.")
        elif cap == "flag_regime":
            sentences.append(
                f"{d['n_flagged']} of {d['n_obs']} observations (a share of "
                f"{d['share_flagged']}) were flagged, in {d['n_episodes']} distinct episodes."
            )
        elif cap == "compare_groups":
            sentences.append(
                f"Comparing flagged with normal periods ({sid}), the mean is "
                f"{d['mean_flagged']} versus {d['mean_normal']} (difference {d['diff_means']}), "
                f"over {d['n_flagged']} flagged and {d['n_normal']} normal observations; this "
                "is descriptive, not a statistical test."
            )
        elif cap == "filter_period":
            sentences.append(
                f"The selected period ({sid}) runs from {d['period_start']} to "
                f"{d['period_end']} and holds {d['n_obs']} observations."
            )
        elif cap == "find_extremes":
            parts = []
            for extreme, word in (("max", "maximum"), ("min", "minimum")):
                if f"{extreme}_date" not in d:
                    continue
                part = f"its {word} was {d[extreme + '_value']} on {d[extreme + '_date']}"
                if artifacts[sid].value[extreme]["tie_count"] > 1:
                    part += f" (the earliest of {d[extreme + '_tie_count']} dates with that value)"
                parts.append(part)
            sentences.append(f"For {_input_name(step, artifacts)}, {' and '.join(parts)}.")
        elif cap == "summary_stats":
            what = _input_name(step, artifacts)
            sentences.append(
                f"Over {d['count']} observations, {what} has mean {d['mean']}, "
                f"standard deviation {d['std']}, minimum {d['min']} and maximum {d['max']} "
                f"(skew {d['skew']}, excess kurtosis {d['excess_kurtosis']})."
            )

    warnings = [i for i in issues if i.severity == "WARN"]
    approved = [i for i in issues if i.severity == "APPROVAL"]
    if warnings:
        sentences.append(
            "Read the results with these warnings in mind: "
            + "; ".join(f"{i.name.replace('_', ' ').lower()} ({i.code})" for i in warnings)
            + "."
        )
    if approved:
        sentences.append(
            "Approved assumptions: "
            + "; ".join(f"{i.name.replace('_', ' ').lower()} ({i.code})" for i in approved)
            + "."
        )
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# LLM narrator and the entry point
# ---------------------------------------------------------------------------


def build_user_prompt(request: str, plan: Plan, facts: list[Fact], issues: list[Issue]) -> str:
    """Assemble the narrator's input: request, plan summary, facts sheet, warnings, approvals."""
    fact_lines = "\n".join(f"- [{f.fact_id}] {f.label}: {f.display}" for f in facts)
    warning_lines = "\n".join(
        f"- {i.code} {i.name}: {i.message}" for i in issues if i.severity == "WARN"
    )
    approval_lines = "\n".join(
        f"- {i.code} {i.name}: {i.message}" for i in issues if i.severity == "APPROVAL"
    )
    return (
        f"<user_request>\n{request}\n</user_request>\n\n"
        f"PLAN:\n{describe_plan(plan)}\n\n"
        f"FACTS SHEET:\n{fact_lines}\n\n"
        f"WARNINGS:\n{warning_lines or '- none'}\n\n"
        f"APPROVED ASSUMPTIONS:\n{approval_lines or '- none'}"
    )


def narrate(
    request: str,
    plan: Plan,
    artifacts: dict[str, Artifact],
    issues: list[Issue],
    mode: str,
    settings: Settings,
    extra_texts: list[str] | None = None,
    client: Any = None,
) -> NarrativeResult:
    """Write the narrative (LLM, or template in mock mode / on failure) and check its numbers.

    Args:
        request: the user's request.
        plan: the executed plan.
        artifacts: step id -> output artifact.
        issues: WARN issues and approved APPROVAL issues to mention.
        mode: "live" or "cached" use the LLM; "mock" uses the template.
        settings: settings (narrator model, effort, tokens, timeout).
        extra_texts: other trusted texts whose numbers are allowed (e.g. dataset titles).
        client: optional OpenAI or Anthropic client (tests inject a fake one).

    Returns:
        A NarrativeResult including the grounding check.
    """
    facts = build_facts_sheet(plan, artifacts)
    result = NarrativeResult(
        text=template_narrative(plan, artifacts, facts, issues),
        source="template",
        prompt_version="template-v1",
        facts=facts,
    )
    if mode != "mock":
        llm_result, error = _llm_narrative(request, plan, facts, issues, settings, client)
        if llm_result is not None:
            result = llm_result
        else:
            result.error = error  # keep the template, record why the LLM was not used

    trusted_texts = [request, *(i.message for i in issues), *(extra_texts or [])]
    result.grounding = check_grounding(result.text, facts, plan, trusted_texts)
    return result


@dataclass
class LLMReply:
    """One text answer from either provider, in the same shape.

    Fields:
        text: the answer text ("" if none).
        model: model that answered.
        input_tokens: prompt tokens reported by the API.
        output_tokens: output tokens reported by the API.
        problem: why the answer cannot be used (refusal, truncation...), or None.
    """

    text: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    problem: str | None


def _ask_openai(prompt: str, settings: Settings, client: Any) -> LLMReply:
    """Ask an OpenAI model for the narrative (Responses API)."""
    response = (client or make_openai_client(settings)).responses.create(
        model=settings.narrator_model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": prompt}],
        reasoning={"effort": settings.narrator_effort},
        max_output_tokens=settings.narrator_max_tokens,  # includes reasoning tokens
    )
    usage = response.usage
    return LLMReply(
        text=response.output_text.strip(),
        model=response.model,
        input_tokens=usage.input_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
        problem=None if response.status == "completed" else f"status {response.status}",
    )


def _ask_anthropic(prompt: str, settings: Settings, client: Any) -> LLMReply:
    """Ask a Claude model for the narrative (Messages API, with refusal fallback)."""
    response = (client or make_anthropic_client(settings)).beta.messages.create(
        model=settings.narrator_model,
        max_tokens=settings.narrator_max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": settings.narrator_effort},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    unusable = response.stop_reason in ("refusal", "max_tokens")
    return LLMReply(
        text=" ".join(b.text for b in response.content if b.type == "text").strip(),
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        problem=f"stop reason {response.stop_reason}" if unusable else None,
    )


def _llm_narrative(
    request: str,
    plan: Plan,
    facts: list[Fact],
    issues: list[Issue],
    settings: Settings,
    client: Any,
) -> tuple[NarrativeResult | None, str | None]:
    """Ask the configured provider's LLM for the narrative.

    Returns:
        (result, None) on success, or (None, error message) if the call failed or
        the answer is unusable.
    """
    prompt = build_user_prompt(request, plan, facts, issues)
    started = time.perf_counter()
    try:
        if settings.llm_provider == "openai":
            reply = _ask_openai(prompt, settings, client)
        else:
            reply = _ask_anthropic(prompt, settings, client)
    except Exception as exc:  # any API failure falls back to the deterministic template
        return None, api_error_message(exc, settings)

    if reply.problem is not None or not reply.text:
        return None, f"LLM narrative unusable ({reply.problem or 'empty answer'})."
    return (
        NarrativeResult(
            text=reply.text,
            source="llm",
            model=reply.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            facts=facts,
        ),
        None,
    )
