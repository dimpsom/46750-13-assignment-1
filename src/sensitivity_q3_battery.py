
"""
Runs a sensitivity analysis for the battery model (Q3_battery).
Run the script with: 
python -m src.sensitivity_q3_battery
Outputs are saved in the `results/Q3_battery/sensitivity` folder.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_question
from src.model_q3 import Question3Model
from src.model_q3_battery import Question3BatteryModel
from src.plotting import plot_schedule


out = Path("results/Q3_battery/sensitivity")
out.mkdir(parents=True, exist_ok=True)

data = load_question("Q3_battery")


def run_sweep(name, values, change_data, xlabel):
    results = []
    schedules = []

    for value in values:
        new_data = change_data(value)

        without = Question3Model(new_data).build().solve()
        with_battery = Question3BatteryModel(new_data).build().solve()

        benefit = with_battery.objective - without.objective
        discharged = with_battery.hourly["discharge"].sum()

        results.append([
            value, benefit, without.objective,
            with_battery.objective, discharged
        ])
        schedules.append(with_battery.hourly)

        print(f"{name}: {value} -> {benefit:.3f} DKK")

    df = pd.DataFrame(
        results,
        columns=[name, "battery_value", "without_battery",
                 "with_battery", "energy_discharged"]
    )
    df.to_csv(out / f"{name}.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(
        df[name], df["battery_value"],
        marker="o", color="midnightblue", label="Battery value"
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Battery value (DKK/day)")

    if name == "price_variation":
        ax2 = ax1.twinx()
        ax2.bar(
            df[name], df["energy_discharged"],
            width=0.2, color="steelblue", alpha=0.4,
            label="Energy discharged"
        )
        ax2.set_ylabel("Energy discharged (kWh/day)")
        ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99))
        ax1.legend(loc="upper left", bbox_to_anchor=(0.35, 0.99))
        
    plt.tight_layout()
    plt.savefig(out / f"{name}.png")
    plt.close()

    return dict(zip(values, schedules)), df


# Capacity
initial_ratio = data.battery_initial_soc_kWh / data.battery_capacity_kWh

capacity, capacity_df = run_sweep(
    "capacity",
    [2, 3, 4, 5, 6, 7, 8],
    lambda x: replace(data,
        battery_capacity_kWh=x,
        battery_initial_soc_kWh=x * initial_ratio),
    "Battery capacity (kWh)"
)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()

for x in [2, 4, 6, 7, 8]:
    ax1.plot(capacity[x].index, capacity[x]["soc"], label=f"{x} kWh")

ax2.plot(
    capacity[4].index,
    data.energy_price,
    color="grey",
    linestyle="--",
    alpha=0.4,
    label="Energy price"
)

ax1.set_xlabel("Hour")
ax1.set_ylabel("State of charge (kWh)")
ax2.set_ylabel("Energy price (DKK/kWh)")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(out / "capacity_soc.png")
plt.close()


# Efficiency
efficiency, efficiency_df = run_sweep(
    "efficiency",
    [0.80, 0.85, 0.90, 0.95, 1.00],
    lambda x: replace(data,
        battery_charging_efficiency=x,
        battery_discharging_efficiency=x),
    "Charging/discharging efficiency"
)

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, sharey=True)

for ax, x in zip(axes, [0.80, 0.95, 1.00]):
    hourly = efficiency[x]
    ax.plot(hourly.index, hourly["charge"], label="Charging")
    ax.plot(hourly.index, -hourly["discharge"], label="Discharging")
    ax.set_title(f"Efficiency = {x:.0%}")
    ax.set_ylabel("Power (kW)")
    ax.legend()

axes[-1].set_xlabel("Hour")
plt.tight_layout()
plt.savefig(out / "efficiency_operation.png")
plt.close()


# Price variation
mean_price = np.mean(data.energy_price)

price, price_df = run_sweep(
    "price_variation",
    [0, 0.5, 1, 1.5, 2],
    lambda x: replace(data,
        energy_price=mean_price + x * (data.energy_price - mean_price)),
    "Price variation factor"
)
