
"""
Sensitivity analysis for Question 3.
Run with:
python -m src.sensitivity_q3
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_question
from src.model_q3 import Question3Model


out = Path("results/Q3/sensitivity")
out.mkdir(parents=True, exist_ok=True)

data = load_question("Q3")


def run_sweep(name, values, change_data, xlabel):
    results = []
    schedules = {}

    for value in values:
        new_data = change_data(value)
        result = Question3Model(new_data).build().solve()

        hourly = result.hourly
        load = hourly["load"].sum()
        imports = hourly["import"].sum()
        exports = hourly["export"].sum()
        deviation = np.sum(
            (hourly["load"].to_numpy() - new_data.reference_load) ** 2
        )

        results.append([
            value, result.objective, load, imports, exports, deviation
        ])
        schedules[value] = hourly

        print(f"{name}: {value} -> {result.objective:.3f} DKK")

    df = pd.DataFrame(
        results,
        columns=[
            name, "objective", "load", "import", "export", "deviation"
        ]
    )
    df.to_csv(out / f"{name}.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    metric = "deviation" if name == "quadratic_disutility" else "load"

    ax1.plot(
        df[name], df[metric],
        marker="o", color="midnightblue",
        label="Squared deviation" if metric == "deviation" else "Daily load"
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(
        "Squared deviation (kWh²)" if metric == "deviation" else "Daily load (kWh)"
    )

    ax2 = ax1.twinx()
    ax2.plot(
        df[name], df["objective"],
        marker="s", linestyle="--",
        color="steelblue", label="Objective"
    )
    ax2.set_ylabel("Objective (DKK/day)")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out / f"{name}.png")
    plt.close()

    return schedules, df


minimum_energy, minimum_energy_df = run_sweep(
    "minimum_energy",
    [20, 25, 30, 35, 40, 45, 50],
    lambda x: replace(data, min_daily_energy_kWh=x),
    "Minimum daily energy (kWh)"
)


quadratic_disutility, quadratic_disutility_df = run_sweep(
    "quadratic_disutility",
    [
        data.quadratic_disutility * x
        for x in [0.25, 0.5, 1, 1.5, 2]
    ],
    lambda x: replace(data, quadratic_disutility=x),
    "Quadratic disutility (DKK/kWh²)"
)


fig, ax = plt.subplots(figsize=(8, 4))

for x in [
    data.quadratic_disutility * factor
    for factor in [0.25, 1, 2]
]:
    hourly = quadratic_disutility[x]
    ax.plot(hourly.index, hourly["load"], label=f"cQ = {x:.2f}")

ax.plot(
    data.hours,
    data.reference_load,
    color="grey",
    linestyle="--",
    label="Reference load"
)

ax.set_xlabel("Hour")
ax.set_ylabel("Load (kWh)")
ax.legend()

plt.tight_layout()
plt.savefig(out / "quadratic_disutility_load.png")
plt.close()
