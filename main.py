"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --scenarios              # also run the example sensitivity scenarios
    python main.py --question Q2_linear --sweep   # Q2.(b): base case + sweep of c_L (both Q2's can be run with/without sweep)
    python main.py --question Q2_quadratic --sweep   # Q2.(c): base case + sweep of c_Q
    python main.py --question Q3            # Q3: quadratic disutility + minimum daily energy

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, Results
from src.model_linear import LinearDisutilityModel, find_breakpoints, plot_c_L_sweep, sweep_c_L
from src.model_quadratic import QuadraticDisutilityModel, check_thresholds, plot_c_Q_sweep, sweep_c_Q
from src.model_q3 import Question3Model
from src.model_q3_battery import Question3BatteryModel
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    if question == "Q3_battery":
        model = Question3BatteryModel(data).build()
    elif question == "Q3":
        model = Question3Model(data).build()
    else:
        model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    if "utility" in results.meta:  # Question 1 metrics (Q3 has no consumption utility)
        print(f"utility:          {results.meta['utility']:.2f} DKK")
        print(f"procurement cost: {results.meta['procurement_cost']:.2f} DKK")
        print(f"net utility:      {results.meta['net_utility']:.2f} DKK")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def run_linear_base_case(out: Path, show: bool) -> Results:
    """Question 2.(b).iii: base case of the linear-disutility model (case Q2_linear)."""
    data = load_question("Q2_linear")
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    results = LinearDisutilityModel(data).build().solve()
    print(results, "\n")
    m = results.meta
    print(f"c_L:                {m['c_L']:.2f} DKK/kWh")
    print(f"procurement cost:   {m['procurement_cost']:.2f} DKK")
    print(f"disutility:         {m['disutility']:.2f} DKK")
    print(f"energy consumed:    {m['energy']:.2f} kWh (reference {data.reference_load.sum():.2f} kWh)")
    print(f"total |deviation|:  {m['abs_deviation']:.2f} kWh")
    print(f"hours at a bound:   {m['n_bound_hours']}")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_linear_sweep(out: Path) -> None:
    """Question 2.(b).iv: sweep of c_L, metrics per run and the values of c_L where the load changes."""
    data = load_question("Q2_linear")
    # grid offset by 0.005 so that no c_L equals one of the (2-decimal) prices exactly -> no ties
    c_grid = np.round(np.arange(0.005, 3.0, 0.01), 3)
    table, loads = sweep_c_L(data, c_grid)
    table.to_csv(out / "Q2_linear_sweep.csv", index=False)

    breakpoints = find_breakpoints(table, loads, data)
    breakpoints.to_csv(out / "Q2_linear_breakpoints.csv", index=False)
    plot_c_L_sweep(table, breakpoints, save_to=out / "sweep_c_L.png")

    print("\nSweep of c_L (every 10th run shown; full table in Q2_linear_sweep.csv):")
    print(table.iloc[::10].round(2).to_string(index=False))
    print("\nValues of c_L at which the optimal load changes:")
    print(breakpoints.round(2).to_string(index=False))
    print(f"\nLargest consumption above the reference in any run: {table['max_dev_up'].max():.4f} kWh")


def run_quadratic_base_case(out: Path, show: bool) -> Results:
    """Question 2.(c).iii: base case of the quadratic-disutility model (case Q2_quadratic)."""
    data = load_question("Q2_quadratic")
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    results = QuadraticDisutilityModel(data).build().solve()
    print(results, "\n")
    m = results.meta
    print(f"c_Q:                {m['c_Q']:.2f} DKK/kWh^2")
    print(f"procurement cost:   {m['procurement_cost']:.2f} DKK")
    print(f"disutility:         {m['disutility']:.2f} DKK")
    print(f"energy consumed:    {m['energy']:.2f} kWh (reference {data.reference_load.sum():.2f} kWh)")
    print(f"total |deviation|:  {m['abs_deviation']:.2f} kWh")
    print(f"hours at a bound:   {m['n_bound_hours']}")
    print(f"hours cut to PV:    {m['n_pv_hours']}")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_quadratic_sweep(out: Path) -> None:
    """Question 2.(c).iv: sweep of c_Q, metrics per run and comparison with the thresholds of Section 2.3.2."""
    data = load_question("Q2_quadratic")
    # log-spaced grid from 0.01 to 10 DKK/kWh^2: the interesting values span two orders of magnitude
    c_grid = np.logspace(-2, 1, 151)
    table, loads = sweep_c_Q(data, c_grid)
    table.to_csv(out / "Q2_quadratic_sweep.csv", index=False)

    thresholds = check_thresholds(data, c_grid, loads)
    thresholds.to_csv(out / "Q2_quadratic_thresholds.csv", index=False)
    plot_c_Q_sweep(table, loads, data, save_to=out / "sweep_c_Q.png")

    print("\nSweep of c_Q (every 10th run shown; full table in Q2_quadratic_sweep.csv):")
    print(table.iloc[::10].round(3).to_string(index=False))
    print("\nThresholds per hour, read from the sweep and predicted (Section 2.3.2):")
    print(thresholds.round(3).to_string(index=False))
    print(f"\nLargest consumption above the reference in any run: {table['max_dev_up'].max():.4f} kWh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--sweep", action="store_true", help="Q2_linear / Q2_quadratic: also sweep the disutility coefficient (c_L or c_Q)")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    if args.question == "Q2_linear":
        run_linear_base_case(out, args.show)
        if args.sweep:
            run_linear_sweep(out)
    elif args.question == "Q2_quadratic":
        run_quadratic_base_case(out, args.show)
        if args.sweep:
            run_quadratic_sweep(out)
    else:
        base = run_base_case(args.question, out, args.show)
        if args.scenarios and base is not None:
            run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()