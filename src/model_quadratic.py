"""Question 2.(c): consumer with a QUADRATIC disutility of deviating from a reference profile.



* the penalty is  - c^Q * (L_t - l_ref_t)^2  (a quadratic term in the objective),

Usage::

    data = load_question("Q2_quadratic")
    results = QuadraticDisutilityModel(data).build().solve()
    results.meta["procurement_cost"], results.meta["disutility"], ...
"""
from __future__ import annotations

from pathlib import Path

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData
from .model import FlexibleConsumerModel, Results


class QuadraticDisutilityModel(FlexibleConsumerModel):
    """Hourly consumption problem with disutility c^Q * (L_t - l_ref_t)^2 (a convex QP)."""

    def __init__(self, data: InputData, c_Q: float | None = None, name: str = "quadratic_disutility",
                 verbose: bool = False):
        """``c_Q`` [DKK/kWh^2] overrides ``data.quadratic_disutility`` (used for the sweep of 2.(c).iv)."""
        super().__init__(data, name=name, verbose=verbose)
        self.c_Q = data.quadratic_disutility if c_Q is None else c_Q

    # ------------------------------------------------------------------ build
    def build(self) -> "QuadraticDisutilityModel":
        d, m, T = self.data, self.m, self.T

        if d.reference_load is None:
            raise ValueError("Question 2 requires a reference load profile.")
        if self.c_Q is None or self.c_Q <= 0:
            raise ValueError("Question 2 (quadratic) requires a quadratic disutility coefficient c_Q > 0.")
        c_Q = self.c_Q
        ref = d.reference_load                      # preferred consumption, kWh/h

        # Effective grid prices
        p_import = d.energy_price + d.import_tariff
        p_export = d.energy_price - d.export_tariff

        # --- Decision variables ------------------------------------------------
        self.var["import"] = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="import")
        self.var["export"] = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="export")
        # load and pv bounds are written as explicit constraints below (so we get their duals)
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="load")
        self.var["pv"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pv")

        imp, exp = self.var["import"], self.var["export"]
        load, pv = self.var["load"], self.var["pv"]

        # --- Objective: maximise  -procurement cost - quadratic disutility ----------
        m.setObjective(
            gp.quicksum(
                - p_import[t] * imp[t]
                - d.pv_marginal_cost * pv[t]
                + p_export[t] * exp[t]
                - c_Q * (load[t] - ref[t]) * (load[t] - ref[t])
                for t in T
            ),
            GRB.MAXIMIZE,
        )

        # --- Constraints (same as Question 1) ---------------------------------------
        # Hourly power balance
        self.con["balance"] = m.addConstrs(
            (load[t] + exp[t] == imp[t] + pv[t] for t in T), name="balance")

        # Load bounds
        self.con["load_min"] = m.addConstrs((load[t] >= d.load_min_kWh for t in T), name="load_min")
        self.con["load_max"] = m.addConstrs((load[t] <= d.load_max_kWh for t in T), name="load_max")

        # PV bounds
        self.con["pv_min"] = m.addConstrs((pv[t] >= 0 for t in T), name="pv_min")
        self.con["pv_max"] = m.addConstrs((pv[t] <= d.pv_available[t] for t in T), name="pv_max")

        m.update()
        return self

    # ------------------------------------------------------------------ solve
    def solve(self) -> Results:
        """Solve, then add the Question 2 metrics to ``results.meta`` (all in DKK or kWh):

        procurement_cost  import cost + PV cost - export revenue
        disutility        c_Q * sum_t (L_t - l_ref_t)^2
        energy            daily energy consumed, sum_t L_t
        abs_deviation     sum_t |L_t - l_ref_t|
        n_bound_hours     hours in which the load is deviating AND sits on L_min or L_max
        n_pv_hours        hours in which the load is cut down exactly to the available PV
        """
        results = super().solve()
        d, hr = self.data, results.hourly
        tol = 1e-5

        p_import, p_export = hr["p_import"], hr["p_export"]
        procurement_cost = float((p_import * hr["import"] + d.pv_marginal_cost * hr["pv"]
                                  - p_export * hr["export"]).sum())
        deviation = hr["load"] - d.reference_load               # kWh, negative = below the reference
        disutility = float(self.c_Q * (deviation ** 2).sum())

        deviating = np.abs(deviation) > tol
        at_bound = (np.abs(hr["load"] - d.load_min_kWh) < tol) | (np.abs(hr["load"] - d.load_max_kWh) < tol)
        at_pv = (np.abs(hr["load"] - d.pv_available) < tol) & (d.reference_load > d.pv_available + tol)

        results.meta.update(
            c_Q=self.c_Q,
            procurement_cost=procurement_cost,
            disutility=disutility,
            net_utility=-(procurement_cost + disutility),
            energy=float(hr["load"].sum()),
            abs_deviation=float(np.abs(deviation).sum()),
            n_bound_hours=int((deviating & at_bound).sum()),
            n_pv_hours=int(at_pv.sum()),
        )
        # objective is the negative of (procurement cost + disutility)
        assert abs(results.meta["net_utility"] - self.m.ObjVal) < 1e-5
        return results


# ====================================================================== 2.(c).iv: sweep of c_Q
def sweep_c_Q(data: InputData, c_grid: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Solve the model once for every value in ``c_grid`` [DKK/kWh^2].

    Returns
    -------
    table : one row per c_Q with procurement_cost [DKK], disutility [DKK], energy [kWh],
            abs_deviation [kWh], n_bound_hours [-], n_pv_hours [-] and max_dev_up [kWh]
            (largest consumption above the reference in any hour).
    loads : array (len(c_grid), 24) with the optimal hourly load [kWh/h] of every run.
    """
    rows, loads = [], []
    for c in c_grid:
        r = QuadraticDisutilityModel(data, c_Q=float(c)).build().solve()
        rows.append({
            "c_Q": float(c),
            "procurement_cost": r.meta["procurement_cost"],
            "disutility": r.meta["disutility"],
            "energy": r.meta["energy"],
            "abs_deviation": r.meta["abs_deviation"],
            "n_bound_hours": r.meta["n_bound_hours"],
            "n_pv_hours": r.meta["n_pv_hours"],
            "max_dev_up": float((r.hourly["load"] - data.reference_load).max()),
        })
        loads.append(r.hourly["load"].to_numpy())
    return pd.DataFrame(rows), np.array(loads)


def check_thresholds(data: InputData, c_grid: np.ndarray, loads: np.ndarray, tol: float = 1e-4) -> pd.DataFrame:
    """Compare the sweep with the thresholds of Section 2.3.2, hour by hour (hours with a reference > 0).

    With  price = min(max(c_PV, p_exp), p_imp)  (the saving per kWh when the cut kWh is PV), the thresholds are

        plateau at PV_max : price / (2 (ref - PV_max))  <=  c_Q  <=  p_imp / (2 (ref - PV_max))   (ref > PV_max only)
        load at L_min     : c_Q <= price / (2 (ref - L_min))

    The columns ``..._sweep`` are read from the sweep (the largest / smallest c_Q on the grid with that behaviour),
    so they are only accurate up to the grid spacing.
    """
    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff
    ref, pvmax = data.reference_load, data.pv_available
    rows = []
    for t in range(data.n_hours):
        if ref[t] <= 0:
            continue
        price = min(max(data.pv_marginal_cost, p_exp[t]), p_imp[t])
        at_min = np.abs(loads[:, t] - data.load_min_kWh) < tol
        row = {
            "hour": t, "ref": ref[t], "pv_max": pvmax[t],
            "Lmin_until_sweep": c_grid[at_min].max() if at_min.any() else np.nan,
            "Lmin_until_pred": price / (2 * (ref[t] - data.load_min_kWh)),
            "plateau_from_sweep": np.nan, "plateau_to_sweep": np.nan,
            "plateau_from_pred": np.nan, "plateau_to_pred": np.nan,
        }
        if ref[t] > pvmax[t] + tol:
            on_pv = np.abs(loads[:, t] - pvmax[t]) < tol
            if on_pv.any():
                row["plateau_from_sweep"], row["plateau_to_sweep"] = c_grid[on_pv].min(), c_grid[on_pv].max()
            row["plateau_from_pred"] = price / (2 * (ref[t] - pvmax[t]))
            row["plateau_to_pred"] = p_imp[t] / (2 * (ref[t] - pvmax[t]))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_c_Q_sweep(table: pd.DataFrame, loads: np.ndarray, data: InputData, hours=(8, 10, 12), save_to=None):
    """Left: costs vs c_Q. Middle: energy and absolute deviation vs c_Q. Right: optimal load of some hours vs c_Q.

    The x-axis is logarithmic because the interesting values of c_Q span two orders of magnitude.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(table["c_Q"], table["procurement_cost"], label="procurement cost [DKK]")
    ax1.plot(table["c_Q"], table["disutility"], label="disutility [DKK]")
    ax1.set(xlabel="c_Q [DKK/kWh$^2$]", ylabel="DKK", title="Costs vs c_Q")
    ax2.plot(table["c_Q"], table["energy"], color="C2", label="energy consumed [kWh]")
    ax2.plot(table["c_Q"], table["abs_deviation"], color="C3", label="total |deviation| [kWh]")
    ax2.set(xlabel="c_Q [DKK/kWh$^2$]", ylabel="kWh", title="Energy and deviation vs c_Q")
    for i, t in enumerate(hours):
        ax3.plot(table["c_Q"], loads[:, t], label=f"hour {t} (ref {data.reference_load[t]:.2f}, PV {data.pv_available[t]:.2f})")
        ax3.axhline(data.pv_available[t], color=f"C{i}", ls=":", lw=0.8)
    ax3.set(xlabel="c_Q [DKK/kWh$^2$]", ylabel="kWh", title="Optimal load of single hours (dotted: PV available)")
    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig