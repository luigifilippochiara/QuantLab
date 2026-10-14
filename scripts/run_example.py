"""End-to-end demo of the first QuantLab slice (no LLM involved yet).

1. Load the sample price series.
2. Load the hand-written example plan (JSON).
3. Validate the plan against the registry.
4. Execute it with the deterministic engine and print the result.
5. Show the guardrails refusing a bad plan and an unknown capability.

Run from the project root with:  uv run python scripts/run_example.py
"""

import json
from pathlib import Path

import pandas as pd

from core.execution import CapabilityRefusedError, execute_plan, run_capability
from core.validation import validate_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = PROJECT_ROOT / "data" / "sample" / "sample_prices.csv"
PLAN_PATH = PROJECT_ROOT / "plans" / "example_rolling_volatility.json"

# The sample data was simulated with this volatility (see make_sample_data.py).
TRUE_ANNUAL_VOL = 0.20


def print_header(title: str) -> None:
    """Print a section title, to make the console output easy to scan."""
    print(f"\n=== {title} ===")


def load_prices(path: Path) -> pd.Series:
    """Load a CSV with `date` and `close` columns into a price Series.

    Args:
        path: path to the CSV file.

    Returns:
        Closing prices as a Series indexed by date, sorted oldest first.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    # Rolling calculations depend on order, so make sure dates are ascending.
    return df.set_index("date")["close"].sort_index()


def show_guardrails(prices: pd.Series, available_inputs: dict) -> None:
    """Show that invalid plans and unknown capabilities are refused.

    Args:
        prices: the sample price series.
        available_inputs: dataset name -> data type, as passed to validate_plan.
    """
    print_header("Guardrails: a bad plan")
    bad_plan = {
        "plan_id": "bad-plan-001",
        "description": "An unknown capability, plus bad parameter values.",
        "steps": [
            {
                "step_id": "s1",
                "capability": "run_python",  # not in the registry
                "input": "prices",
                "params": {"code": "print('hello')"},
                "output": "anything",
            },
            {
                "step_id": "s2",
                "capability": "rolling_volatility",
                "input": "prices",
                # window is out of range; periods_per_year is a string, not an int.
                "params": {"window": 1000, "periods_per_year": "252"},
                "output": "vol_bad",
            },
        ],
    }
    for error in validate_plan(bad_plan, available_inputs):
        print(f"  - {error}")

    print_header("Guardrails: calling the engine directly, skipping validation")
    try:
        run_capability("run_python", prices, {"code": "print('hello')"})
    except CapabilityRefusedError as exc:
        print(f"  Refused: {exc}")


def main() -> None:
    """Run the demo: load data and plan, validate, execute, print results."""
    print_header("Data")
    prices = load_prices(PRICES_PATH)
    print(
        f"Loaded {len(prices)} daily prices, "
        f"{prices.index[0].date()} to {prices.index[-1].date()}"
    )

    print_header("Plan")
    plan = json.loads(PLAN_PATH.read_text())
    print(json.dumps(plan, indent=2))

    print_header("Validation")
    # Tell the validator which datasets the plan may use, and their types.
    available_inputs = {"prices": "price_series"}
    errors = validate_plan(plan, available_inputs)
    if errors:
        print("Plan is INVALID:")
        for error in errors:
            print(f"  - {error}")
        return  # never execute an invalid plan
    print("OK: plan is valid.")

    print_header("Execution")
    results = execute_plan(plan, datasets={"prices": prices})
    vol = results["vol_20d"]
    print(f"Output 'vol_20d': {len(vol)} values (first one on {vol.index[0].date()})")
    print("\nLast 10 values:")
    print(vol.tail(10).round(4).to_string())

    print("\nSummary:")
    print(f"  latest: {vol.iloc[-1]:.4f}")
    print(f"  mean:   {vol.mean():.4f}")
    print(f"  range:  {vol.min():.4f} to {vol.max():.4f}")
    # Sanity check against the known answer. A 20-return window is a small sample,
    # so estimates scatter noticeably around the true value.
    print(f"  (true simulated volatility: {TRUE_ANNUAL_VOL:.2f})")

    show_guardrails(prices, available_inputs)


if __name__ == "__main__":
    main()
