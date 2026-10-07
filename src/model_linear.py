"""Question 2.(b): consumer with a LINEAR disutility of deviating from a reference profile.

The model follows the same build() / solve() pattern as ``FlexibleConsumerModel`` in
``src/model.py``. Compared with Question 1 two things change:

* the constant utility u^L * L_t is replaced by the penalty  - c^L * |L_t - l_ref_t|,
* the absolute value is linearised with two non-negative auxiliary variables
      L_t = l_ref_t + dL_up_t - dL_dn_t ,     |L_t - l_ref_t| = dL_up_t + dL_dn_t .

Usage::

    data = load_question("Q2_linear")
    results = LinearDisutilityModel(data).build().solve()
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


class LinearDisutilityModel(FlexibleConsumerModel):
    """Hourly consumption problem with disutility c^L * |L_t - l_ref_t| (an LP)."""

    def __init__(self, data: InputData, c_L: float | None = None, name: str = "linear_disutility",
                 verbose: bool = False):
        """``c_L`` [DKK/kWh] overrides ``data.linear_disutility`` (used for the sweep of 2.(b).iv)."""
        super().__init__(data, name=name, verbose=verbose)
        self.c_L = data.linear_disutility if c_L is None else c_L

    # ------------------------------------------------------------------ build
    def build(self) -> "LinearDisutilityModel":
        d, m, T = self.data, self.m, self.T

        if d.reference_load is None:
            raise ValueError("Question 2 requires a reference load profile.")
        if self.c_L is None or self.c_L < 0:
            raise ValueError("Question 2 (linear) requires a linear disutility coefficient c_L >= 0.")
        c_L = self.c_L
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
        # auxiliary variables: deviation above / below the reference profile (kWh)
        self.var["dev_up"] = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="dev_up")
        self.var["dev_dn"] = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="dev_dn")

        imp, exp = self.var["import"], self.var["export"]
        load, pv = self.var["load"], self.var["pv"]
        dev_up, dev_dn = self.var["dev_up"], self.var["dev_dn"]

        # --- Objective: maximise  -procurement cost - disutility ------------------
        m.setObjective(
            gp.quicksum(
                - p_import[t] * imp[t]
                - d.pv_marginal_cost * pv[t]
                + p_export[t] * exp[t]
                - c_L * (dev_up[t] + dev_dn[t])
                for t in T
            ),
            GRB.MAXIMIZE,
        )

        # --- Constraints -------------------------------------------------------
        # Hourly power balance
        self.con["balance"] = m.addConstrs(
            (load[t] + exp[t] == imp[t] + pv[t] for t in T), name="balance")

        # Definition of the load through the deviation variables
        self.con["deviation"] = m.addConstrs(
            (load[t] == ref[t] + dev_up[t] - dev_dn[t] for t in T), name="deviation")

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
        disutility        c_L * sum_t |L_t - l_ref_t|
        energy            daily energy consumed, sum_t L_t
        abs_deviation     sum_t |L_t - l_ref_t|
        n_bound_hours     hours in which the load is deviating AND sits on L_min or L_max
        """
        results = super().solve()
        d, hr = self.data, results.hourly
        tol = 1e-6

        p_import, p_export = hr["p_import"], hr["p_export"]
        procurement_cost = float((p_import * hr["import"] + d.pv_marginal_cost * hr["pv"]
                                  - p_export * hr["export"]).sum())
        abs_dev = float((hr["dev_up"] + hr["dev_dn"]).sum())
        disutility = self.c_L * abs_dev

        deviating = (hr["dev_up"] + hr["dev_dn"]) > tol
        at_bound = (np.abs(hr["load"] - d.load_min_kWh) < tol) | (np.abs(hr["load"] - d.load_max_kWh) < tol)

        results.meta.update(
            c_L=self.c_L,
            procurement_cost=procurement_cost,
            disutility=disutility,
            net_utility=-(procurement_cost + disutility),
            energy=float(hr["load"].sum()),
            abs_deviation=abs_dev,
            n_bound_hours=int((deviating & at_bound).sum()),
        )
        # objective is the negative of (procurement cost + disutility)
        assert abs(results.meta["net_utility"] - self.m.ObjVal) < 1e-6
        return results


# ====================================================================== 2.(b).iv: sweep of c_L
def sweep_c_L(data: InputData, c_grid: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Solve the model once for every value in ``c_grid`` [DKK/kWh].

    Returns
    -------
    table : one row per c_L with procurement_cost [DKK], disutility [DKK], energy [kWh],
            abs_deviation [kWh], n_bound_hours [-] and max_dev_up [kWh] (largest consumption
            above the reference in any hour).
    loads : array (len(c_grid), 24) with the optimal hourly load [kWh/h] of every run.
    """
    rows, loads = [], []
    for c in c_grid:
        r = LinearDisutilityModel(data, c_L=float(c)).build().solve()
        rows.append({
            "c_L": float(c),
            "procurement_cost": r.meta["procurement_cost"],
            "disutility": r.meta["disutility"],
            "energy": r.meta["energy"],
            "abs_deviation": r.meta["abs_deviation"],
            "n_bound_hours": r.meta["n_bound_hours"],
            "max_dev_up": float(r.hourly["dev_up"].max()),
        })
        loads.append(r.hourly["load"].to_numpy())
    return pd.DataFrame(rows), np.array(loads)


def find_breakpoints(table: pd.DataFrame, loads: np.ndarray, data: InputData, tol: float = 1e-6) -> pd.DataFrame:
    """Values of c_L at which the optimal load profile changes, and the price they match.

    A breakpoint lies between two consecutive grid points whose load profiles differ; its value
    is taken as the midpoint (so use a grid offset from the 2-decimal prices, e.g. 0.005, 0.015, ...).
    The column ``matches`` lists the effective import price (p_imp), effective export price (p_exp)
    or PV marginal cost (c_PV) that equals the breakpoint.
    """
    # candidate thresholds: c_PV and the effective prices of every hour
    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff
    cands = [("c_PV", data.pv_marginal_cost)]
    cands += [(f"p_imp[{t}]", p_imp[t]) for t in range(data.n_hours)]
    cands += [(f"p_exp[{t}]", p_exp[t]) for t in range(data.n_hours)]

    rows = []
    c = table["c_L"].to_numpy()
    for i in range(1, len(c)):
        changed = np.abs(loads[i] - loads[i - 1]) > tol
        if not changed.any():
            continue
        thr = round((c[i - 1] + c[i]) / 2, 3)
        half = (c[i] - c[i - 1]) / 2 + 1e-9
        matches = [name for name, v in cands if abs(v - thr) <= half]
        rows.append({
            "c_L_threshold": thr,
            "energy_before": table["energy"].iloc[i - 1],
            "energy_after": table["energy"].iloc[i],
            "hours_changed": ",".join(str(t) for t in np.where(changed)[0]),
            "matches": ", ".join(matches),
        })
    return pd.DataFrame(rows)


def plot_c_L_sweep(table: pd.DataFrame, breakpoints: pd.DataFrame, save_to=None):
    """Metrics of the c_L sweep (cost, disutility, energy, absolute deviation) with the breakpoints marked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(table["c_L"], table["procurement_cost"], label="procurement cost [DKK]")
    ax1.plot(table["c_L"], table["disutility"], label="disutility [DKK]")
    ax1.set(xlabel="c_L [DKK/kWh]", ylabel="DKK", title="Costs vs c_L")
    ax2.plot(table["c_L"], table["energy"], color="C2", label="energy consumed [kWh]")
    ax2.plot(table["c_L"], table["abs_deviation"], color="C3", label="total |deviation| [kWh]")
    ax2.set(xlabel="c_L [DKK/kWh]", ylabel="kWh", title="Energy and deviation vs c_L")
    for ax in (ax1, ax2):
        for thr in breakpoints["c_L_threshold"]:
            ax.axvline(thr, color="grey", ls=":", lw=0.8)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig
