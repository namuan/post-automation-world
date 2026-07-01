#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "persistent-cache@git+https://github.com/namuan/persistent-cache"
# ]
# ///
"""
Agent-based stock-flow simulation inspired by arXiv:2606.20649v1.

Usage:
./post_automation_sim.py -h
./post_automation_sim.py --scenario foreign_ai_untaxed --plot-output results.csv
./post_automation_sim.py --compare --seeds 0 1 2 3 4
./post_automation_sim.py -vv --scenario full_toolkit --periods 600 --agents 2000
"""
from __future__ import annotations

import logging
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging(verbosity):
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    logging.captureWarnings(capture=True)


@dataclass(frozen=True)
class Params:
    agents: int = 2000
    periods: int = 600
    seed: int = 0

    productivity: float = 1.0
    k0_per_capita: float = 5.0
    cognitive_share: float = 0.50
    robot_capital_share0: float = 0.50
    init_wealth_sigma: float = 1.15

    e_top: float = 1.20
    e_routine: float = 0.60
    e_cognitive: float = 0.60
    theta_cognitive: float = 0.50
    a_r_base: float = 0.50
    a_ai_base: float = 0.50
    robot_ramp_start: float = 60.0
    robot_ramp_speed: float = 0.05
    robot_ramp_size: float = 0.45
    ai_ramp_start: float = 110.0
    ai_ramp_speed: float = 0.10
    ai_ramp_size: float = 0.49

    depreciation_robot: float = 0.05
    depreciation_ai: float = 0.18
    invest_damping: float = 0.25
    min_capital: float = 1e-9

    c_income: float = 0.85
    c_profit: float = 0.35
    c_wealth: float = 0.03
    gov_cost_share: float = 0.0
    debt_rate: float = 0.02

    ret_sigma: float = 0.05
    ret_persist: float = 0.92
    demographic_reset: float = 0.02

    corporate_tax: float = 0.0
    income_tax: float = 0.0
    wealth_tax: float = 0.0
    progressive_wealth_tax: bool = False
    ubi_share: float = 0.0
    citizens_fund: bool = False

    avoidance_elasticity: float = 0.75
    migration_semi_elast: float = 0.02
    offshore_adjust_speed: float = 0.20

    foreign_owned_ip_share: float = 1.0
    foreign_equity_share0: float = 0.0
    mu_frac: float = 0.25
    s_home: float = 1.0
    robot_tax: float = 0.0
    dst_ai: float = 0.0
    tax_repat: float = 0.0
    rebate_repat_tax: bool = False
    repatriation_share: float = 0.0
    ai_supply_elasticity: float = 0.0
    contestability: float = 0.0
    profit_shift_elasticity: float = 0.0
    robot_ip: bool = False

    reinstatement: float = 0.0
    unemployment_pass_through: float = 0.0
    benefit_replacement: float = 0.50


SCENARIOS: dict[str, Params] = {
    "laissez_faire": Params(),
    "wealth_tax": Params(wealth_tax=0.02),
    "progressive_wealth_tax": Params(wealth_tax=0.015, progressive_wealth_tax=True),
    "income_tax_ubi": Params(income_tax=0.30, ubi_share=0.12, gov_cost_share=0.02),
    "thin_welfare_state": Params(income_tax=0.15, ubi_share=0.12, gov_cost_share=0.10),
    "domestic_owner": Params(foreign_owned_ip_share=0.0, income_tax=0.15),
    "domestic_owner_wealth_tax": Params(foreign_owned_ip_share=0.0, income_tax=0.15, wealth_tax=0.05),
    "foreign_ai_untaxed": Params(foreign_owned_ip_share=1.0, foreign_equity_share0=0.10),
    "robot_tax": Params(foreign_owned_ip_share=1.0, foreign_equity_share0=0.10, robot_tax=0.15),
    "dst_ai": Params(foreign_owned_ip_share=1.0, foreign_equity_share0=0.10, dst_ai=0.10),
    "withholding_rebate": Params(
        foreign_owned_ip_share=1.0,
        foreign_equity_share0=0.10,
        tax_repat=0.30,
        rebate_repat_tax=True,
    ),
    "compute_offshore": Params(foreign_owned_ip_share=1.0, foreign_equity_share0=0.10, s_home=0.20),
    "full_toolkit": Params(
        foreign_owned_ip_share=1.0,
        foreign_equity_share0=0.10,
        robot_tax=0.15,
        dst_ai=0.10,
        tax_repat=0.30,
        s_home=1.0,
        wealth_tax=0.01,
    ),
}


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="foreign_ai_untaxed")
    parser.add_argument("--compare", action="store_true", help="Run all named scenarios")
    parser.add_argument("--agents", type=int, help="Override number of agents")
    parser.add_argument("--periods", type=int, help="Override number of periods")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Monte Carlo seeds")
    parser.add_argument("--plot-output", type=Path, help="Write period-level time series CSV")
    parser.add_argument("--summary-output", type=Path, help="Write terminal summary CSV")
    parser.add_argument("--mu-frac", type=float, help="Override AI rent share of cognitive value added")
    parser.add_argument("--wealth-tax", type=float, help="Override wealth tax rate")
    parser.add_argument("--dst-ai", type=float, help="Override digital-services levy rate")
    parser.add_argument("--tax-repat", type=float, help="Override rent withholding rate")
    parser.add_argument("--robot-tax", type=float, help="Override robot tax rate")
    parser.add_argument("--repatriation-share", type=float, help="Rent share taken as goods, not reinvested")
    parser.add_argument("--ai-supply-elasticity", type=float, help="Elasticity of AI deployment to net rent")
    parser.add_argument("--profit-shift-elasticity", type=float, help="Profit-shifting response to rent tax wedge")
    parser.add_argument("--contestability", type=float, help="Share of AI rent competed away")
    return parser.parse_args()


def with_overrides(params: Params, args) -> Params:
    updates = {}
    for field, attr in [
        ("agents", "agents"),
        ("periods", "periods"),
        ("mu_frac", "mu_frac"),
        ("wealth_tax", "wealth_tax"),
        ("dst_ai", "dst_ai"),
        ("tax_repat", "tax_repat"),
        ("robot_tax", "robot_tax"),
        ("repatriation_share", "repatriation_share"),
        ("ai_supply_elasticity", "ai_supply_elasticity"),
        ("profit_shift_elasticity", "profit_shift_elasticity"),
        ("contestability", "contestability"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            updates[field] = value
    return replace(params, **updates)


def logistic(t: int, start: float, speed: float) -> float:
    z = np.clip(-speed * (t - start), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


def automation_paths(t: int, p: Params) -> tuple[float, float]:
    a_r = p.a_r_base + p.robot_ramp_size * logistic(t, p.robot_ramp_start, p.robot_ramp_speed)
    a_ai = p.a_ai_base + p.ai_ramp_size * logistic(t, p.ai_ramp_start, p.ai_ramp_speed)
    return min(a_r, 0.995), min(a_ai, 0.995)


def ces_output_and_shares(x_l: float, x_k: float, automation: float, elasticity: float) -> tuple[float, float, float]:
    x_l = max(x_l, 1e-12)
    x_k = max(x_k, 1e-12)
    rho = (elasticity - 1.0) / elasticity
    coeff_l = (1.0 - automation) ** (1.0 / elasticity)
    coeff_k = automation ** (1.0 / elasticity)
    terms = coeff_l * x_l**rho + coeff_k * x_k**rho
    y = terms ** (1.0 / rho)
    labour_share = coeff_l * x_l**rho / terms
    capital_share = coeff_k * x_k**rho / terms
    return y, labour_share, capital_share


def production(k_robot: float, k_ai: float, t: int, p: Params) -> dict[str, float]:
    a_r, a_ai = automation_paths(t, p)
    total_l = p.agents
    l_cog = total_l * p.cognitive_share
    l_routine = total_l - l_cog

    yr, sr_l, sr_k = ces_output_and_shares(l_routine, k_robot, a_r, p.e_routine)
    yc, sc_l, sc_k = ces_output_and_shares(l_cog, k_ai, a_ai, p.e_cognitive)

    rho_top = (p.e_top - 1.0) / p.e_top
    coeff_r = (1.0 - p.theta_cognitive) ** (1.0 / p.e_top)
    coeff_c = p.theta_cognitive ** (1.0 / p.e_top)
    top_terms = coeff_r * yr**rho_top + coeff_c * yc**rho_top
    y = p.productivity * top_terms ** (1.0 / rho_top)
    share_r = coeff_r * yr**rho_top / top_terms
    share_c = coeff_c * yc**rho_top / top_terms

    value_r = y * share_r
    value_c = y * share_c
    gross_mu = p.mu_frac * max(0.0, 1.0 - p.contestability)
    tax_wedge = min(0.95, p.dst_ai + p.tax_repat)
    supply_scale = max(0.0, (1.0 - tax_wedge) ** p.ai_supply_elasticity)
    ai_rent = gross_mu * supply_scale * value_c * p.foreign_owned_ip_share
    robot_ip_rent = gross_mu * supply_scale * value_r * p.foreign_owned_ip_share if p.robot_ip else 0.0

    routine_factor_base = max(0.0, value_r - robot_ip_rent)
    cognitive_factor_base = max(0.0, value_c - ai_rent)
    wage_routine = routine_factor_base * sr_l
    robot_income = routine_factor_base * sr_k
    wage_cognitive = cognitive_factor_base * sc_l
    ai_compute_income = cognitive_factor_base * sc_k
    labour_income = wage_routine + wage_cognitive
    capital_income = robot_income + ai_compute_income

    return {
        "Y": y,
        "a_robot": a_r,
        "a_ai": a_ai,
        "value_routine": value_r,
        "value_cognitive": value_c,
        "wage_routine": wage_routine,
        "wage_cognitive": wage_cognitive,
        "labour_income": labour_income,
        "robot_income": robot_income,
        "ai_compute_income": ai_compute_income,
        "capital_income": capital_income,
        "ai_rent": ai_rent,
        "robot_ip_rent": robot_ip_rent,
        "labour_share": labour_income / y if y else 0.0,
        "capital_share": capital_income / y if y else 0.0,
        "rent_share": (ai_rent + robot_ip_rent) / y if y else 0.0,
    }


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total <= 0:
        return 0.0
    sorted_values = np.sort(values)
    n = values.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - ((n + 1.0) / n))


def top_share(values: np.ndarray, pct: float) -> float:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = values.sum()
    if total <= 0:
        return 0.0
    n_top = max(1, int(np.ceil(values.size * pct)))
    return float(np.sort(values)[-n_top:].sum() / total)


def allocate_top_weighted_move(wealth: np.ndarray, target_move: float) -> np.ndarray:
    move = np.zeros_like(wealth)
    remaining = min(target_move, wealth.sum())
    if remaining <= 0:
        return move
    for idx in np.argsort(wealth)[::-1]:
        amount = min(wealth[idx], remaining)
        move[idx] = amount
        remaining -= amount
        if remaining <= 1e-9:
            break
    return move


def progressive_rates(wealth: np.ndarray, base_rate: float) -> np.ndarray:
    p90, p99 = np.quantile(wealth, [0.90, 0.99])
    rates = np.full_like(wealth, base_rate)
    rates[wealth >= p90] = base_rate * 2.0
    rates[wealth >= p99] = base_rate * 4.0
    return np.clip(rates, 0.0, 0.15)


def run_simulation(params: Params, scenario: str = "custom") -> pd.DataFrame:
    rng = np.random.default_rng(params.seed)
    n = params.agents
    k_total = params.k0_per_capita * n
    k_robot = k_total * params.robot_capital_share0
    k_ai = k_total - k_robot

    foreign_equity = k_total * params.foreign_equity_share0
    gov_equity = 0.0
    household_equity_total = k_total - foreign_equity - gov_equity
    initial = rng.lognormal(mean=0.0, sigma=params.init_wealth_sigma, size=n)
    household_equity = initial / initial.sum() * household_equity_total
    offshore_equity = np.zeros(n)
    return_type = rng.normal(0.0, params.ret_sigma, size=n)
    is_cognitive = np.zeros(n, dtype=bool)
    is_cognitive[rng.choice(n, size=int(n * params.cognitive_share), replace=False)] = True

    gov_deposits = 0.0
    foreign_deposits = 0.0
    records = []

    for t in range(params.periods):
        prod = production(k_robot, k_ai, t, params)
        y = prod["Y"]
        capital_income = prod["capital_income"]

        robot_tax = params.robot_tax * prod["robot_income"]
        corporate_base = max(0.0, capital_income - robot_tax)
        corporate_tax = params.corporate_tax * corporate_base

        rent = prod["ai_rent"] + prod["robot_ip_rent"]
        shifted_rent_share = min(0.90, params.profit_shift_elasticity * (params.dst_ai + params.tax_repat))
        reachable_rent = rent * (1.0 - shifted_rent_share)
        dst_tax = params.dst_ai * (prod["value_cognitive"] + reachable_rent)
        withholding_tax = params.tax_repat * reachable_rent
        after_tax_rent = max(0.0, rent - dst_tax - withholding_tax)

        domestic_capital_income = max(0.0, capital_income - robot_tax - corporate_tax)
        ownership_base = max(k_total, 1e-12)
        household_capital_income = domestic_capital_income * household_equity.sum() / ownership_base
        gov_capital_income = domestic_capital_income * gov_equity / ownership_base
        foreign_capital_income = domestic_capital_income * foreign_equity / ownership_base

        wages = np.where(
            is_cognitive,
            prod["wage_cognitive"] / max(1, is_cognitive.sum()),
            prod["wage_routine"] / max(1, (~is_cognitive).sum()),
        )
        if params.reinstatement > 0.0:
            restored = params.reinstatement * (prod["a_ai"] - params.a_ai_base)
            wages *= 1.0 + max(0.0, restored)

        disp_robot = max(0.0, prod["a_robot"] - params.a_r_base)
        disp_ai = max(0.0, prod["a_ai"] - params.a_ai_base)
        unemployment_rate = min(
            0.95,
            params.unemployment_pass_through
            * ((1.0 - params.cognitive_share) * disp_robot + params.cognitive_share * disp_ai)
            * (1.0 - params.reinstatement),
        )
        employed = np.ones(n, dtype=bool)
        if unemployment_rate > 0:
            n_unemployed = int(unemployment_rate * n)
            skill_order = np.argsort(wages + rng.normal(0, 1e-9, size=n))
            employed[skill_order[:n_unemployed]] = False
            lost_wages = wages[~employed].sum()
            wages[~employed] = 0.0
            if employed.any():
                wages[employed] += lost_wages / employed.sum()
        avg_employed_wage = wages[employed].mean() if employed.any() else 0.0
        benefits = np.where(~employed, params.benefit_replacement * avg_employed_wage, 0.0)

        if household_equity.sum() > 0 and household_capital_income > 0:
            return_type = (
                params.ret_persist * return_type
                + np.sqrt(max(0.0, 1.0 - params.ret_persist**2))
                * rng.normal(0.0, params.ret_sigma, size=n)
            )
            weights = household_equity * np.exp(return_type)
            household_capital = weights / weights.sum() * household_capital_income
        else:
            household_capital = np.zeros(n)

        income_tax = params.income_tax * (wages + household_capital)
        ubi = np.full(n, params.ubi_share * y / n)
        if params.citizens_fund and gov_capital_income > 0:
            ubi += gov_capital_income / n
            gov_capital_income = 0.0
        if params.rebate_repat_tax and withholding_tax > 0:
            ubi += withholding_tax / n

        if params.wealth_tax > 0:
            rates = (
                progressive_rates(household_equity, params.wealth_tax)
                if params.progressive_wealth_tax
                else np.full(n, params.wealth_tax)
            )
            taxable_base = household_equity * np.power(np.clip(1.0 - rates, 1e-9, 1.0), params.avoidance_elasticity)
            wealth_tax = rates * taxable_base
        else:
            wealth_tax = np.zeros(n)
        wealth_tax_total = wealth_tax.sum()
        household_equity = np.maximum(0.0, household_equity - wealth_tax)
        gov_equity += wealth_tax_total

        target_offshore = min(0.50, params.migration_semi_elast * params.wealth_tax * 100.0) * (
            household_equity.sum() + offshore_equity.sum()
        )
        offshore_gap = max(0.0, target_offshore - offshore_equity.sum())
        offshore_move = allocate_top_weighted_move(household_equity, params.offshore_adjust_speed * offshore_gap)
        household_equity -= offshore_move
        offshore_equity += offshore_move
        foreign_equity += offshore_move.sum()

        disposable = wages + household_capital + benefits + ubi - income_tax
        consumption = (
            params.c_income * np.maximum(0.0, wages + benefits + ubi - income_tax)
            + params.c_profit * np.maximum(0.0, household_capital)
            + params.c_wealth * np.maximum(0.0, household_equity)
        )
        consumption = np.minimum(consumption, np.maximum(0.0, disposable + 0.08 * household_equity))
        household_saving = disposable - consumption
        household_equity = np.maximum(0.0, household_equity + household_saving)

        demographic = rng.random(n) < params.demographic_reset
        if demographic.any():
            estate = household_equity[demographic].sum()
            household_equity[demographic] = 0.0
            household_equity += estate / n
            offshore_estate = offshore_equity[demographic].sum()
            offshore_equity[demographic] = 0.0
            offshore_equity += offshore_estate / n

        gov_spending = params.gov_cost_share * y + ubi.sum() + benefits.sum()
        gov_revenue = (
            income_tax.sum()
            + wealth_tax_total
            + corporate_tax
            + robot_tax
            + dst_tax
            + (0.0 if params.rebate_repat_tax else withholding_tax)
        )
        gov_primary_balance = gov_revenue + gov_capital_income - gov_spending
        gov_deposits = gov_deposits * (1.0 + params.debt_rate) + gov_primary_balance

        foreign_reinvest = after_tax_rent * (1.0 - params.repatriation_share) + foreign_capital_income
        foreign_deposits += after_tax_rent * params.repatriation_share
        if foreign_reinvest > 0:
            foreign_equity += foreign_reinvest

        aggregate_consumption = consumption.sum()
        gross_saving = max(0.0, y - aggregate_consumption - gov_spending - after_tax_rent * params.repatriation_share)
        k_robot_after_dep = max(params.min_capital, k_robot * (1.0 - params.depreciation_robot))
        k_ai_after_dep = max(params.min_capital, k_ai * (1.0 - params.depreciation_ai))
        ret_robot = prod["robot_income"] / max(k_robot, params.min_capital) - params.depreciation_robot
        ret_ai = prod["ai_compute_income"] / max(k_ai, params.min_capital) - params.depreciation_ai
        target_ai_share = 1.0 / (1.0 + np.exp(np.clip(-8.0 * (ret_ai - ret_robot), -60, 60)))
        current_ai_share = k_ai / max(k_total, params.min_capital)
        invest_ai_share = np.clip(
            current_ai_share + params.invest_damping * (target_ai_share - current_ai_share),
            0.05,
            0.95,
        )
        k_robot = k_robot_after_dep + gross_saving * (1.0 - invest_ai_share)
        k_ai = k_ai_after_dep + gross_saving * invest_ai_share
        k_total = k_robot + k_ai

        claims_total = household_equity.sum() + gov_equity + foreign_equity
        if claims_total <= 0:
            household_equity += k_total / n
            claims_total = k_total
        scale = k_total / claims_total
        household_equity *= scale
        offshore_equity *= scale
        gov_equity *= scale
        foreign_equity *= scale

        measured_wealth = household_equity
        true_wealth = household_equity + offshore_equity
        govt_net_worth = gov_equity + gov_deposits
        foreign_net_worth = foreign_equity + foreign_deposits
        deposits_sum = gov_deposits + foreign_deposits - gov_deposits - foreign_deposits
        net_worth_sum = household_equity.sum() + gov_equity + foreign_equity

        record = {
            "scenario": scenario,
            "seed": params.seed,
            "period": t,
            "output": y,
            "output_per_capita": y / n,
            "capital": k_total,
            "capital_output_ratio": k_total / y if y else np.nan,
            "k_robot": k_robot,
            "k_ai": k_ai,
            "a_robot": prod["a_robot"],
            "a_ai": prod["a_ai"],
            "labour_share": prod["labour_share"],
            "capital_share": prod["capital_share"],
            "rent_share": prod["rent_share"],
            "ai_rent_pct_output": prod["ai_rent"] / y if y else 0.0,
            "capital_sector_revenue_pct_output": (corporate_tax + robot_tax + dst_tax + withholding_tax) / y
            if y
            else 0.0,
            "tax_revenue_pct_output": gov_revenue / y if y else 0.0,
            "gov_net_worth_output": govt_net_worth / y if y else np.nan,
            "foreign_net_worth_output": foreign_net_worth / y if y else np.nan,
            "foreign_ownership": foreign_equity / k_total if k_total else 0.0,
            "state_ownership": gov_equity / k_total if k_total else 0.0,
            "household_ownership": household_equity.sum() / k_total if k_total else 0.0,
            "offshore_household_share": offshore_equity.sum() / max(true_wealth.sum(), 1e-12),
            "wealth_gini": gini(measured_wealth),
            "true_wealth_gini": gini(true_wealth),
            "top_1_share": top_share(true_wealth, 0.01),
            "top_10_share": top_share(true_wealth, 0.10),
            "unemployment_rate": unemployment_rate,
            "accounting_equity_gap": net_worth_sum - k_total,
            "accounting_deposit_gap": deposits_sum,
        }
        if t % 50 == 0:
            logging.debug("period=%s %s", t, {k: round(v, 4) for k, v in record.items() if isinstance(v, float)})
        records.append(record)

    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame, tail_periods: int = 40) -> pd.DataFrame:
    rows = []
    group_cols = ["scenario", "seed"]
    for (scenario, seed), group in df.groupby(group_cols, sort=False):
        tail = group.tail(tail_periods)
        terminal = group.iloc[-1]
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "output_per_capita": tail["output_per_capita"].mean(),
                "capital_output_ratio": tail["capital_output_ratio"].mean(),
                "wealth_gini": tail["wealth_gini"].mean(),
                "true_wealth_gini": tail["true_wealth_gini"].mean(),
                "top_1_share": tail["top_1_share"].mean(),
                "top_10_share": tail["top_10_share"].mean(),
                "ai_rent_pct_output": tail["ai_rent_pct_output"].mean(),
                "capital_sector_revenue_pct_output": tail["capital_sector_revenue_pct_output"].mean(),
                "foreign_ownership": terminal["foreign_ownership"],
                "state_ownership": terminal["state_ownership"],
                "gov_net_worth_output": terminal["gov_net_worth_output"],
                "unemployment_rate": tail["unemployment_rate"].mean(),
                "max_abs_equity_gap": group["accounting_equity_gap"].abs().max(),
                "max_abs_deposit_gap": group["accounting_deposit_gap"].abs().max(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    numeric = summary.select_dtypes(include=[np.number]).columns.difference(["seed"])
    means = summary.groupby("scenario", sort=False)[numeric].mean()
    stds = summary.groupby("scenario", sort=False)[numeric].std(ddof=0).add_suffix("_sd")
    return pd.concat([means, stds], axis=1).reset_index()


def main(args):
    logging.debug(f"This is a debug log message: {args.verbose}")
    logging.info(f"This is an info log message: {args.verbose}")

    scenario_names = sorted(SCENARIOS) if args.compare else [args.scenario]
    frames = []
    for scenario_name in scenario_names:
        for seed in args.seeds:
            params = with_overrides(replace(SCENARIOS[scenario_name], seed=seed), args)
            logging.info("Running %s seed=%s params=%s", scenario_name, seed, asdict(params))
            frames.append(run_simulation(params, scenario=scenario_name))

    results = pd.concat(frames, ignore_index=True)
    summary = summarize(results)
    aggregate = aggregate_summary(summary)

    display_cols = [
        "scenario",
        "wealth_gini",
        "true_wealth_gini",
        "ai_rent_pct_output",
        "capital_sector_revenue_pct_output",
        "foreign_ownership",
        "gov_net_worth_output",
        "capital_output_ratio",
    ]
    print(aggregate[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    if args.plot_output:
        results.to_csv(args.plot_output, index=False)
        logging.warning("Wrote time series to %s", args.plot_output)
    if args.summary_output:
        aggregate.to_csv(args.summary_output, index=False)
        logging.warning("Wrote summary to %s", args.summary_output)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
