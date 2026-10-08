from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from .model import FlexibleConsumerModel


class Question3BatteryModel(FlexibleConsumerModel):
    """Question 3: quadratic disutility + minimum daily energy requirement."""

    def build(self) -> "Question3BatteryModel":
        d, m, T = self.data, self.m, self.T

        # Check that the required Q3 data are available
        if d.quadratic_disutility is None:
            raise ValueError("Q3 requires quadratic_disutility.")

        if d.reference_load is None:
            raise ValueError("Q3 requires reference_load.")

        if d.min_daily_energy_kWh is None:
            raise ValueError("Q3 requires min_daily_energy_kWh.")

        # Effective grid prices
        p_import = d.energy_price + d.import_tariff
        p_export = d.energy_price - d.export_tariff

        # ---------------------------------------------------------------
        # Decision variables
        # ---------------------------------------------------------------

        self.var["import"] = m.addVars(
            T, lb=0.0, vtype=GRB.CONTINUOUS, name="import"
        )

        self.var["export"] = m.addVars(
            T, lb=0.0, vtype=GRB.CONTINUOUS, name="export"
        )

        # Explicit bounds are added below so that duals can be extracted
        self.var["load"] = m.addVars(
            T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="load"
        )

        self.var["pv"] = m.addVars(
            T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pv"
        )

        imp = self.var["import"]
        exp = self.var["export"]
        load = self.var["load"]
        pv = self.var["pv"]

        # NEW Battery related variables
        self.var["charge"] = m.addVars(
            T, lb=0, ub=d.battery_max_charge_kW, vtype=GRB.CONTINUOUS, name="charge"
        )

        self.var["discharge"] = m.addVars(
            T, lb=0, ub=d.battery_max_discharge_kW, vtype=GRB.CONTINUOUS, name="discharge"
        )

        self.var["soc"] = m.addVars(
            range(len(T)+1), lb=0, ub=d.battery_capacity_kWh, vtype=GRB.CONTINUOUS, name="soc"
        )

        ch = self.var["charge"]
        dis = self.var["discharge"]
        soc = self.var["soc"]

        # ---------------------------------------------------------------
        # Objective: Question 2(c) quadratic disutility
        # ---------------------------------------------------------------

        m.setObjective(
            gp.quicksum(
                p_export[t] * exp[t]
                - p_import[t] * imp[t]
                - d.pv_marginal_cost * pv[t]
                - d.quadratic_disutility
                * (load[t] - d.reference_load[t]) ** 2
                for t in T
            ),
            GRB.MAXIMIZE,
        )

        # ---------------------------------------------------------------
        # Constraints
        # ---------------------------------------------------------------

        # Hourly power balance
        self.con["balance"] = m.addConstrs(
            (
                load[t] + exp[t] + ch[t] == imp[t] + pv[t] + dis[t]
                for t in T
            ),
            name="balance",
        )

        # Load bounds
        self.con["load_min"] = m.addConstrs(
            (load[t] >= d.load_min_kWh for t in T),
            name="load_min",
        )

        self.con["load_max"] = m.addConstrs(
            (load[t] <= d.load_max_kWh for t in T),
            name="load_max",
        )

        # PV bounds
        self.con["pv_min"] = m.addConstrs(
            (pv[t] >= 0.0 for t in T),
            name="pv_min",
        )

        self.con["pv_max"] = m.addConstrs(
            (pv[t] <= d.pv_available[t] for t in T),
            name="pv_max",
        )

        # NEW constraint for Question 3:
        # minimum total daily energy consumption
        self.con["minimum_daily_energy"] = m.addConstr(
            gp.quicksum(load[t] for t in T)
            >= d.min_daily_energy_kWh,
            name="minimum_daily_energy",
        )

        # NEW constraints for battery state of charge "soc"
        # Sate of charge for each timestep (hour)
        self.con["soc_balance"] = m.addConstrs(
            (
                soc[t+1] == soc[t] + d.battery_charging_efficiency * ch[t] - dis[t] / d.battery_discharging_efficiency
            for t in T
            ),
            name = "soc_balance",
        )

        # Initial and final states of charge
        self.con["soc_initial"] = m.addConstr(
            soc[0] == d.battery_initial_soc_kWh,
            name="soc_initial",
        )
        
        return self
