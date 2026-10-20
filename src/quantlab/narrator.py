"""Narrator: a short plain-English summary of the results, with every number checked.

Role in the architecture: this is the second (and last) module that talks to
an LLM. The narrator never sees data series. It gets a FACTS SHEET: the key
numbers the executor computed, each already rounded and formatted. It is told
to use only those numbers, exactly as displayed, and to do no arithmetic.

Then `check_grounding` verifies that claim deterministically: it extracts every
number from the narrative and checks that each one is a correct rounding of a
computed fact (or its x100 percentage form), a plan parameter, a number from the
request or a warning, an annualisation factor, or (only when a significance test
ran) the 5% significance level and 95% confidence level. Dates are facts too:
every date in the narrative must be a date of the facts sheet.

That check is guardrail N01 UNGROUNDED_NARRATIVE (action WARN), the narrator
check: an LLM narrative that fails it is withheld (kept in the run record, never
shown as the summary) and the deterministic template narrative is shown instead,
with an N01 issue listing the numbers that failed. The run's numerical results
are not affected, and the results tables are always shown as the source of truth.

Without an API key (mock mode), or if the LLM call fails, the deterministic
template narrator writes the summary; its numbers are checked the same way.
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
from quantlab.guardrails import PERIOD_NAMES, Action, Issue, make_issue
from quantlab.mock_scenarios import match_scenario
from quantlab.plan import Plan, describe_plan
from quantlab.planner import (
    FALLBACK_BETA,
    api_error_message,
    make_anthropic_client,
    make_openai_client,
)
from quantlab.types import ANNUALISATION_FACTORS, Artifact, DataType, TokenUsage
from quantlab.validator import fill_defaults

NARRATOR_PROMPT_VERSION = "narrator-v7"
# The narratives an LLM wrote in live runs and mock mode replays (Scenario.narrative).
RECORDED_NARRATIVES_VERSION = "recorded-narratives-v1"

# The note shown directly under the narrative (app and CLI) and kept in the run
# record: who wrote it and, only when the grounding check passed, that its numbers
# were checked. narrative_note() decides which sentences apply.
AI_WRITTEN_NOTE = (
    "This interpretation was written by an AI and may contain errors in wording or reasoning."
)
NUMBERS_CHECKED_NOTE = (
    "Every number in it has been checked automatically against the computed results (N01)."
)

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
6. Describe what was computed; do not claim causes or forecasts. Never say that a flag causes \
or predicts returns: a comparison or test shows an association in this sample, nothing more.
7. Claim statistical significance only from a significance test (hac_mean_difference); \
compare_groups is descriptive. The test is two-sided: it asks whether the outcome differs \
between flagged and other periods, not which is higher. First answer that plainly: does it \
differ, and is the difference statistically significant. Then say which group's mean was \
higher, giving both means, as a description of the sample. Then state the difference in \
means and its p-value, and always the number of distinct flagged episodes. If the p-value \
is 0.05 or more, write "no statistically significant difference" (never "no difference"); \
below 0.05, write "a statistically significant difference at the 5% level".
8. If DATASET NOTES say how a series is quoted, every direction you state (for example which \
currency strengthened) must follow them.
9. Write 3 to 6 sentences as one paragraph. No headings, no bullet points.
"""

# A number in text: digits with optional thousands separators and decimals.
# The look-behind skips digits glued to letters or dots, such as step ids (s7),
# guardrail codes (E04), dataset ids (DGS10) and release names (H.15).
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
# The conventional significance level used to read p-values, and the matching
# confidence level of the reported interval (hac_mean_difference's ci_low/ci_high).
SIGNIFICANCE_LEVEL = 0.05
CONFIDENCE_LEVEL = 0.95

DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{MONTH}\s*,?\s+(\d{{4}})\b", re.I)
MONTH_DAY_YEAR = re.compile(rf"\b{MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(\d{{4}})\b", re.I)
MONTH_YEAR = re.compile(rf"\b{MONTH}\s+(\d{{4}})\b", re.I)


class Fact(BaseModel):
    """One computed number (or date) the narrator may use.

    Fields:
        fact_id: stable id, e.g. "s7.mean_flagged".
        label: what it is, e.g. "s7 compare_groups mean flagged".
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
        source: "llm", "mock" (an LLM narrative recorded from a live run, replayed in
            mock mode) or "template".
        model: model that wrote it (None for the template).
        prompt_version: narrator prompt version.
        latency_ms: time spent writing it.
        input_tokens: prompt tokens reported by the API.
        output_tokens: output tokens reported by the API.
        error: why the LLM was not used, if it failed (or was withheld) and the
            template took over.
        facts: the facts sheet it was given.
        grounding: the result of the numeric grounding check (None until checked).
        withheld: the LLM narrative withheld by N01 because it failed the grounding
            check, with its own grounding result (kept for audit; never shown as
            the summary), or None.
        issues: N01 UNGROUNDED_NARRATIVE issues (empty when every number is grounded).
        note: the note shown under the narrative (see narrative_note); "" if none applies.
        usage: the tokens of every narrator call made for this narrative, whether its
            text is shown, withheld or was unusable (zero in mock mode). Set on the
            narrative narrate() returns; None on a withheld one.
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
    withheld: "NarrativeResult | None" = None
    issues: list[Issue] = Field(default_factory=list)
    note: str = ""
    usage: TokenUsage | None = None


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
        elif artifact.data_type == DataType.FLAGS:
            flags = artifact.value
            add(step.id, cap, "n_obs", meta.n_obs)
            add(step.id, cap, "n_flagged", int(flags.sum()))
            add(step.id, cap, "share_flagged", float(flags.mean()))
            add(step.id, cap, "n_episodes", count_episodes(flags))
        else:  # TABLE: named scalars
            for key, value in artifact.value.items():
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
    `extra_texts` (the request, warning messages, dataset titles), the
    annualisation factors and, when the facts include a p-value, the significance
    and confidence levels (0.05 and 0.95, and 5 and 95). Dates are facts: a date
    (YYYY-MM-DD, or written in words such as "19 November 2021") is grounded only
    if it is a date of the facts sheet, and a month ("November 2021") only if a
    fact date falls in it.
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
    if any(fact.fact_id.endswith(".p_value") for fact in facts):
        # "Not significant at the 5% level", "95% confidence interval": accepted only
        # when a test ran, so they cannot slip into a descriptive narrative.
        allowed += [SIGNIFICANCE_LEVEL, SIGNIFICANCE_LEVEL * 100]
        allowed += [CONFIDENCE_LEVEL, CONFIDENCE_LEVEL * 100]
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


def deterministic_narrative(
    plan: Plan,
    artifacts: dict[str, Artifact],
    facts: list[Fact],
    issues: list[Issue],
    dataset_notes: dict[str, str] | None = None,
) -> str:
    """Write a deterministic summary from the facts sheet (used in mock mode and as fallback).

    Args:
        plan: the executed plan.
        artifacts: step id -> output artifact.
        facts: the facts sheet (all numbers are taken from its display strings).
        issues: warnings and approved assumptions to mention.
        dataset_notes: dataset id -> its quote convention, for the datasets used that
            have one (see describe_quotes).

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
    # Which way an exchange rate moves when a currency strengthens, so its signed
    # returns below can be read the right way round.
    sentences += describe_quotes(dataset_notes)

    for step in plan.steps:
        if step.id not in artifacts:
            continue
        sid, cap = step.id, step.capability
        # d: this step's facts by short name, e.g. d["mean_flagged"] == "0.002093".
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
                f"{d['mean_flagged']} versus {d['mean_normal']} (difference {d['diff_means']}; "
                f"standard deviation {d['std_flagged']} versus {d['std_normal']}), over "
                f"{d['n_flagged']} flagged observations in {d['n_episodes_flagged']} distinct "
                f"episodes and {d['n_normal']} normal observations; this is descriptive, not a "
                "statistical test."
            )
        elif cap == "hac_mean_difference":
            # Read the p-value from the exact computed value, not its rounded display.
            significant = artifacts[sid].value["p_value"] < SIGNIFICANCE_LEVEL
            verdict = (
                "a statistically significant difference at the 5% level"
                if significant
                else "no statistically significant difference at the 5% level"
            )
            # The answer first, in plain words. The test is two-sided: it asks whether the
            # means DIFFER, not which is higher, so that is what the first sentence says.
            # Which group's mean is higher is then described from the two means (the
            # sign of the exact difference), not claimed as what the test showed.
            difference = artifacts[sid].value["diff_means"]
            period = PERIOD_NAMES.get(artifacts[sid].meta.frequency, "period")
            if "horizon" in d:  # forward returns: the period after each flag
                subject, when = f"next-{period} returns", "after"
            else:  # returns of the flagged periods themselves
                subject, when = "returns", "in"
            sentences.append(
                f"The test asks whether {subject} {when} flagged {period}s differ from those "
                f"{when} the other {period}s (two-sided): the difference is"
                f"{'' if significant else ' not'} statistically significant at the 5% level."
            )
            if difference == 0:
                sentences.append(f"The two means are equal: {d['mean_flagged']}.")
            else:
                higher = f"flagged {period}s" if difference > 0 else f"the other {period}s"
                sentences.append(
                    f"The mean was higher {when} {higher}: {d['mean_flagged']} {when} flagged "
                    f"{period}s against {d['mean_normal']} {when} the others."
                )
            outcome = f"{d['horizon']}-period forward returns" if "horizon" in d else "returns"
            sentences.append(
                f"The significance test ({sid}) finds {verdict} in {outcome} between flagged "
                f"and normal periods: the difference in means is {d['diff_means']}, with a "
                f"Newey-West standard error of {d['se_hac']} ({d['maxlags_used']} lags), a "
                f"p-value of {d['p_value']} and a 95% confidence interval from {d['ci_low']} "
                f"to {d['ci_high']}."
            )
            sentences.append(
                f"The {d['n_flagged']} flagged observations (against {d['n_normal']} normal "
                f"ones) form {d['n_episodes_flagged']} distinct episodes, the real measure of "
                "independent evidence because flagged periods cluster; the test describes an "
                "association in this sample, not a cause or a forecast."
            )

    warnings = [i for i in issues if i.action == Action.WARN]
    approved = [i for i in issues if i.action == Action.APPROVE]
    if warnings:
        sentences.append(
            "Read the results with these warnings in mind: "
            + "; ".join(f"{i.name.replace('_', ' ').lower()} ({i.code})" for i in warnings)
            + "."
        )
    if approved:
        # The step id tells apart the same assumption approved on two steps (e.g. the
        # return method of compute_returns and of forward_return).
        sentences.append(
            "Approved assumptions: "
            + "; ".join(
                f"{i.name.replace('_', ' ').lower()} ({i.code}"
                + (f", {i.step_id})" if i.step_id else ")")
                for i in approved
            )
            + "."
        )
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# LLM narrator and the entry point
# ---------------------------------------------------------------------------


def describe_quotes(dataset_notes: dict[str, str] | None) -> list[str]:
    """One sentence per quote convention, e.g. "DEXSZUS is quoted as USD/CHF, in ...".

    Args:
        dataset_notes: dataset id -> its quote convention (data/catalog.json), or None.

    Returns:
        The sentences, in the order given; none without notes.
    """
    notes = dataset_notes or {}
    return [f"{dataset_id} is quoted as {note}" for dataset_id, note in notes.items()]


def build_user_prompt(
    request: str,
    plan: Plan,
    facts: list[Fact],
    issues: list[Issue],
    dataset_notes: dict[str, str] | None = None,
) -> str:
    """Assemble the narrator's input: request, plan, dataset notes, facts, warnings, approvals."""
    fact_lines = "\n".join(f"- [{f.fact_id}] {f.label}: {f.display}" for f in facts)
    warning_lines = "\n".join(
        f"- {i.code} {i.name}: {i.message}" for i in issues if i.action == Action.WARN
    )
    approval_lines = "\n".join(
        f"- {i.code} {i.name}: {i.message}" for i in issues if i.action == Action.APPROVE
    )
    note_lines = "\n".join(f"- {sentence}" for sentence in describe_quotes(dataset_notes))
    return (
        f"<user_request>\n{request}\n</user_request>\n\n"
        f"PLAN:\n{describe_plan(plan)}\n\n"
        f"DATASET NOTES:\n{note_lines or '- none'}\n\n"
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
    dataset_notes: dict[str, str] | None = None,
    client: Any = None,
    data_sha256: dict[str, str] | None = None,
) -> NarrativeResult:
    """Write the narrative (LLM, recorded or template) and check its numbers.

    Args:
        request: the user's request.
        plan: the executed plan.
        artifacts: step id -> output artifact.
        issues: WARN issues and approved APPROVE issues to mention.
        mode: "live" or "cached" use the LLM; "mock" replays the narrative recorded
            for an unedited example workflow (recorded_narrative), else the template.
        settings: settings (narrator model, effort, tokens, timeout).
        extra_texts: other trusted texts whose numbers are allowed (e.g. dataset titles).
        dataset_notes: dataset id -> quote convention, for the datasets used that have
            one: the template states it, and the LLM is told to follow it.
        client: optional OpenAI or Anthropic client (tests inject a fake one).
        data_sha256: dataset id -> SHA-256 of each dataset the run loaded; a recorded
            narrative is replayed only on the data it was written from.

    Returns:
        A NarrativeResult including the grounding check and any N01 issues. An
        LLM or recorded narrative that fails the check is in `withheld`, and the
        template is the narrative returned.
    """
    facts = build_facts_sheet(plan, artifacts)
    trusted_texts = [request, *(i.message for i in issues), *(extra_texts or [])]
    template = NarrativeResult(
        text=deterministic_narrative(plan, artifacts, facts, issues, dataset_notes),
        source="template",
        prompt_version="template-v1",
        facts=facts,
    )
    template.grounding = check_grounding(template.text, facts, plan, trusted_texts)
    result = template
    candidate, usage = None, TokenUsage()  # mock mode calls no LLM
    if mode == "mock":
        recorded = recorded_narrative(request, plan, data_sha256 or {})
        if recorded is not None:
            candidate = NarrativeResult(
                text=recorded,
                source="mock",
                prompt_version=RECORDED_NARRATIVES_VERSION,
                facts=facts,
            )
    else:
        candidate, error, usage = _llm_narrative(
            request, plan, facts, issues, dataset_notes, settings, client
        )
        if candidate is None:
            template.error = error  # keep the template, record why the LLM was not used
    if candidate is not None:
        # The same check for a live and a recorded narrative.
        candidate.grounding = check_grounding(candidate.text, facts, plan, trusted_texts)
        if candidate.grounding.grounded:
            result = candidate
        else:
            # N01: a narrative quoting a number the run did not produce is never
            # shown as the summary; the checked template replaces it.
            template.withheld = candidate
            written = "recorded" if candidate.source == "mock" else "LLM"
            template.error = f"The {written} narrative was withheld by N01 UNGROUNDED_NARRATIVE."
    result.issues = check_narrative_grounding(result)
    result.note = narrative_note(result)
    result.usage = usage
    return result


def recorded_narrative(request: str, plan: Plan, data_sha256: dict[str, str]) -> str | None:
    """The narrative recorded for an example workflow, when this run is that example unedited.

    An example's canned plan may come with a narrative an LLM wrote for it in a live
    run (mock_scenarios.Scenario.narrative). It describes that run's results only, so
    it is used only when the request matches the example, the plan ran exactly as
    canned (the same steps, inputs and parameters, defaults filled in) and on the same
    data files. After any edit, or on other data, the numbers differ and the template
    is written instead.

    Args:
        request: the user's request.
        plan: the executed plan (defaults filled in).
        data_sha256: dataset id -> SHA-256 of each dataset the run loaded.

    Returns:
        The recorded text, or None.
    """
    scenario = match_scenario(request)
    if scenario is None or scenario.narrative is None:
        return None
    canned = fill_defaults(Plan(plan_id="canned", request=request, **scenario.plan))

    def shape(p: Plan) -> list[tuple[Any, ...]]:
        return [(s.id, s.capability, s.inputs, s.params) for s in p.steps]

    same_data = data_sha256 == scenario.narrative.dataset_sha256
    return scenario.narrative.text if same_data and shape(plan) == shape(canned) else None


def narrative_note(result: NarrativeResult) -> str:
    """The note shown under a narrative: who wrote it, and whether its numbers were checked.

    The AI sentence applies only to a narrative an LLM wrote (live, or recorded from a
    live run and replayed in mock mode): the template is not written by an AI. The
    numbers sentence is added only when the grounding check passed and raised no N01
    issue, so the note never claims the numbers were verified when they were not
    (after an N01, the issue is shown instead).

    Args:
        result: the narrative, with its grounding checked and its N01 issues set.

    Returns:
        The note, or "" when neither sentence applies.
    """
    sentences = []
    if result.source in ("llm", "mock"):
        sentences.append(AI_WRITTEN_NOTE)
    if result.grounding is not None and result.grounding.grounded and not result.issues:
        sentences.append(NUMBERS_CHECKED_NOTE)
    return " ".join(sentences)


def check_narrative_grounding(result: NarrativeResult) -> list[Issue]:
    """N01 UNGROUNDED_NARRATIVE: the narrative quotes numbers the run did not produce.

    Raised when the LLM's narrative was withheld because check_grounding found
    numbers or dates in it that match no value the run computed or used, and, as
    a safety net, if the narrative shown (the deterministic template) ever fails
    the same check. Without it, an invented or misquoted figure could reach the
    reader looking as trustworthy as the computed results. The action is WARN:
    the numerical results are not affected, so the run still completes.

    Args:
        result: the narrative, with its grounding checked.

    Returns:
        The N01 issues (empty when every number is grounded).
    """
    issues = []
    if result.withheld is not None:
        ungrounded = ", ".join(result.withheld.grounding.ungrounded)
        issues.append(
            make_issue(
                "N01",
                None,
                f"The LLM narrative quotes numbers that match no value of the run: {ungrounded}. "
                "It was withheld, and the deterministic summary is shown instead.",
                "Rely on the results and the facts sheet; the withheld text is kept in the "
                "run record.",
                key="llm",
            )
        )
    if not result.grounding.grounded:
        ungrounded = ", ".join(result.grounding.ungrounded)
        issues.append(
            make_issue(
                "N01",
                None,
                f"The deterministic summary quotes numbers that match no value of the run: "
                f"{ungrounded}.",
                "Rely on the results tables and the facts sheet, which come straight from "
                "the executor.",
                key="template",
            )
        )
    return issues


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
    dataset_notes: dict[str, str] | None,
    settings: Settings,
    client: Any,
) -> tuple[NarrativeResult | None, str | None, TokenUsage]:
    """Ask the configured provider's LLM for the narrative.

    Returns:
        (result, None, usage) on success, or (None, error message, usage) if the
        call failed or the answer is unusable. `usage` holds the tokens the API
        reported, even for an unusable answer (it was billed); None counts if the
        call failed before answering.
    """
    prompt = build_user_prompt(request, plan, facts, issues, dataset_notes)
    started = time.perf_counter()
    try:
        if settings.llm_provider == "openai":
            reply = _ask_openai(prompt, settings, client)
        else:
            reply = _ask_anthropic(prompt, settings, client)
    except Exception as exc:  # any API failure falls back to the deterministic template
        no_answer = TokenUsage(model=settings.narrator_model, input_tokens=None, output_tokens=None)
        return None, api_error_message(exc, settings), no_answer

    usage = TokenUsage(
        model=settings.narrator_model,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
    )
    if reply.problem is not None or not reply.text:
        return None, f"LLM narrative unusable ({reply.problem or 'empty answer'}).", usage
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
        usage,
    )
