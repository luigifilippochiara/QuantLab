"""Command-line interface.

    uv run python -m quantlab.cli demo                    # the example workflows, mock mode
    uv run python -m quantlab.cli run "<request>" --datasets NASDAQCOM --mode mock --approve-all
    uv run python -m quantlab.cli replay <run_id>          # re-run without the LLM, compare hashes
    uv run python -m quantlab.cli trace <run_id> <step_id> # where does this number come from?
    uv run python -m quantlab.cli eval --mode mock         # planner evaluation report
    uv run python -m quantlab.cli eval --mode live --case s1_volatility_flag_success

Role in the architecture: a thin, headless front end over engine.py,
lineage.py and evaluation.py; it contains no analysis logic of its own.
"""

import argparse
import sys
from collections.abc import Sequence

from quantlab.config import LLM_MODES, Settings, load_settings
from quantlab.engine import Session, run_request
from quantlab.evaluation import run_evaluation, write_report
from quantlab.lineage import TraceResult, load_run, replay, trace
from quantlab.mock_scenarios import SCENARIOS
from quantlab.plan import describe_plan


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Print rows as a plain-text table with aligned columns."""
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    for index, row in enumerate(cells):
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
        if index == 0:
            print("  ".join("-" * width for width in widths))


def all_issue_codes(session: Session) -> list[str]:
    """Sorted codes of every issue raised in a session: planner refusal, validator,
    executor and narrator check."""
    codes = set(session.report.codes()) if session.report else set()
    if session.execution:
        codes |= {issue.code for issue in session.execution.issues}
    if session.narrative:
        codes |= {issue.code for issue in session.narrative.issues}
    return sorted(codes)


def print_session(session: Session) -> None:
    """Print a run: plan, issues, approvals, results, narrative and where it was saved."""
    print(f"Run {session.run_id}  (LLM mode: {session.mode})")
    if session.plan is not None:
        print("\nPlan:")
        for line in describe_plan(session.plan).splitlines():
            print(f"  {line}")
        if session.plan.planner_notes:
            print(f"  planner notes (unverified): {session.plan.planner_notes}")
    issues = list(session.report.issues) if session.report else []
    if session.execution:
        issues += [i for i in session.execution.issues if i not in issues]
    if session.narrative:
        issues += session.narrative.issues
    if issues:
        print("\nIssues (action, code, enforced by):")
        for issue in issues:
            print(f"  [{issue.action}] {issue.code} {issue.issue_id} "
                  f"({issue.enforced_by}, {issue.type}): "
                  f"{issue.message}")  # fmt: skip
            if issue.suggestion:
                print(f"      -> {issue.suggestion}")
    for approval in session.approvals:
        print(f"  approved {approval.issue_id} by {approval.approved_by} at {approval.approved_at}")

    print(f"\nStatus: {session.status}")
    reason = session.build_record().error
    if reason:
        print(f"  {reason}")
    if session.narrative is not None:
        print("\nResults (facts sheet):")
        by_step: dict[str, list[str]] = {}
        for fact in session.narrative.facts:
            step_id, key = fact.fact_id.split(".", 1)
            by_step.setdefault(step_id, []).append(f"{key}={fact.display}")
        for step in session.plan.steps:
            if step.id in by_step:
                print(f"  {step.id} {step.capability}: " + " ".join(by_step[step.id]))
        grounding = session.narrative.grounding
        badge = (
            "all numbers verified"
            if grounding.grounded
            else f"UNGROUNDED numbers: {grounding.ungrounded}"
        )
        if session.narrative.withheld:
            badge += "; the LLM narrative was withheld (N01)"
        print(f"\nNarrative ({session.narrative.source}; {badge}):")
        print(f"  {session.narrative.text}")
        if session.narrative.note:
            print(f"  Note: {session.narrative.note}")
    if session.manifest_path:
        print(f"\nManifest: {session.manifest_path}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace, settings: Settings) -> int:
    """Run example workflows 1-5 headlessly, then print example 1's trace (the audit trail).

    Only scenarios 1 and 2 are auto-approved; blocked scenarios stay blocked. In
    mock mode the plans are the canned ones, so both the status and the issue
    codes must match. With a real LLM (live/cached) only the status is checked:
    the LLM may legitimately block for a different reason, e.g. refuse scenario 5
    outright (P01) instead of proposing a plan that the validator blocks (V08).
    Running it in live mode also fills the plan cache for later cached-mode sessions.
    """
    rows = []
    all_ok = True
    sessions = {}
    for scenario in SCENARIOS:
        session = run_request(
            scenario.request,
            list(scenario.dataset_ids),
            mode=args.mode,
            approve=scenario.auto_approve,
            settings=settings,
        )
        sessions[scenario.id] = session
        codes = all_issue_codes(session)
        ok = session.status == scenario.expected_status
        if args.mode == "mock":
            ok = ok and set(scenario.expected_codes) <= set(codes)
        all_ok &= ok
        rows.append(
            [scenario.id, session.status, ",".join(codes) or "-", session.run_id,
             "ok" if ok else "MISMATCH"]
        )  # fmt: skip
    print_table(["scenario", "status", "issue codes", "run_id", "check"], rows)
    if args.mode != "mock":
        print(f"\n(LLM: {settings.llm_provider} / {settings.planner_model}; only the outcome "
              "status is checked, the issue codes show how each block happened.)")  # fmt: skip
    print("\nAll scenarios behaved as expected." if all_ok else "\nSome scenarios did NOT.")

    # Scenario 6 is the audit trail of scenario 1: trace its last step (the
    # significance test) back to the snapshot file.
    first = sessions["volatility_flag_success"]
    if first.status == "completed":
        print("\nScenario 6 (traceability): ", end="")
        print_trace(first.trace(first.plan.steps[-1].id))
    return 0 if all_ok else 1


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    """Run one request through the whole pipeline."""
    session = run_request(
        args.request, args.datasets, mode=args.mode, approve=args.approve_all, settings=settings
    )
    print_session(session)
    if session.unresolved_approvals() and not args.approve_all:
        print("\nNothing was executed: re-run with --approve-all to approve the issues above.")
    return 0 if session.status == "completed" else 1


def cmd_replay(args: argparse.Namespace, settings: Settings) -> int:
    """Replay a recorded run without the LLM and compare every output hash."""
    report = replay(args.run_id, settings)
    print(f"Replay of {report.run_id}: original status {report.original_status}, "
          f"replay status {report.replay_status}")  # fmt: skip
    if report.datasets:
        print("\nDatasets:")
        print_table(
            ["dataset", "recorded sha256", "current sha256", "match"],
            [
                [c.dataset_id, c.recorded_sha256[:16], (c.current_sha256 or "missing")[:16],
                 "yes" if c.match else "NO"]
                for c in report.datasets
            ],
        )  # fmt: skip
    if report.steps:
        print("\nSteps:")
        print_table(
            ["step", "code", "capability", "original hash", "replay hash", "match"],
            [
                [s.step_id, s.capability_code or "-", s.capability, (s.original_hash or "-")[:16],
                 (s.replay_hash or "-")[:16], "yes" if s.match else "NO"]
                for s in report.steps
            ],
        )  # fmt: skip
    print(f"\n{report.message}")
    print("ALL HASHES MATCH" if report.all_match else "MISMATCH")
    return 0 if report.all_match else 1


def print_trace(result: TraceResult) -> None:
    """Print a lineage chain: each step back to the source files."""
    print(f"lineage of {result.step_id} in run {result.run_id}:\n")
    for step in result.chain:
        inputs = ", ".join(f"{k}<-{v}" for k, v in step.inputs.items()) or "-"
        print(f"{step.step_id} {step.capability_code or '-'} {step.capability}  inputs: {inputs}")
        print(f"    params:  {step.params}")
        if step.edited:
            print(f"    edited by the user (planner proposed): {step.edited}")
        print(f"    library: {step.library_calls}")
        print(f"    output:  {step.n_obs} obs {step.start}..{step.end}  sha256 {step.output_hash}")
    print("\nSources:")
    for source in result.sources:
        print(f"  {source.dataset_id} (FRED {source.fred_id}) sha256 {source.sha256}")
        print(f"    retrieved {source.retrieved_at}, used {source.start}..{source.end}, "
              f"{source.rows_used} rows")  # fmt: skip


def cmd_trace(args: argparse.Namespace, settings: Settings) -> int:
    """Print the lineage of one step back to the source files."""
    print_trace(trace(load_run(args.run_id, settings.runs_dir), args.step_id))
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    """Evaluate the planner on the golden requests and write a Markdown report.

    Item checks (expected versus planned) are printed for the cases named with
    --case, all of them; without --case, only the items that do not match.
    """
    report = run_evaluation(args.mode, settings, case_ids=args.cases)
    path = write_report(report)
    print_table(
        ["case", "expect", "correct", "planned / issues"],
        [
            [s.case_id, s.expect, "yes" if s.correct else "NO",
             f"{len(s.planned)} steps; {','.join(s.issue_codes) or '-'}"]
            for s in report.scores
        ],
    )  # fmt: skip
    for s in report.scores:
        shown = s.checks if args.cases else [c for c in s.checks if not c.match]
        if shown:
            print(f"\n{s.case_id}, item by item:")
            print_table(
                ["item", "expected", "planned", "match"],
                [[c.item, c.expected, c.actual, "yes" if c.match else "NO"] for c in shown],
            )
    print("\nMetrics:")
    for name, value in report.metrics.items():
        print(f"  {name}: {'n/a' if value is None else value}")
    print(f"\nReport written to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m quantlab.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="run example workflows 1-5, then trace example 1")
    demo.add_argument("--mode", choices=LLM_MODES, default="mock")

    run = commands.add_parser("run", help="run one request")
    run.add_argument("request")
    run.add_argument("--datasets", nargs="+", required=True)
    run.add_argument("--mode", choices=LLM_MODES, default=None, help="default: from .env")
    run.add_argument("--approve-all", action="store_true", help="approve every APPROVE issue")

    replay_cmd = commands.add_parser("replay", help="replay a run without the LLM")
    replay_cmd.add_argument("run_id")

    trace_cmd = commands.add_parser("trace", help="lineage of a step back to the source files")
    trace_cmd.add_argument("run_id")
    trace_cmd.add_argument("step_id")

    evaluate = commands.add_parser("eval", help="evaluate the planner on golden requests")
    evaluate.add_argument("--mode", choices=LLM_MODES, default="mock")
    evaluate.add_argument(
        "--case", action="append", dest="cases", help="evaluate only this golden case (repeatable)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run a subcommand.

    Args:
        argv: arguments (default: sys.argv[1:]).

    Returns:
        Process exit code: 0 on success / expected outcome, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    settings = load_settings()
    handlers = {
        "demo": cmd_demo,
        "run": cmd_run,
        "replay": cmd_replay,
        "trace": cmd_trace,
        "eval": cmd_eval,
    }
    return handlers[args.command](args, settings)


if __name__ == "__main__":
    sys.exit(main())
