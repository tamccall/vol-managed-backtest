import math

import bt
import numpy as np
import pandas as pd


class TSDWeights(bt.algos.Algo):
    def __call__(self, target):
        date = target.now
        tmp_weight = {}
        for k, v in self.weights.items():
            weight = v.at[date]
            tmp_weight[k] = weight if not math.isnan(weight) else 0

        target.temp["weights"] = tmp_weight
        return True

    def __init__(self, **weights):
        super().__init__("TSDWeights")
        self.weights = weights


treasuries = pd.read_csv("daily-treasury-rates.csv", index_col=0)
treasuries = treasuries.set_index(pd.to_datetime(treasuries.index)).sort_index() / 100
two_year = treasuries["2 Yr"]


def vol_managed_potfolio_vix(data, c, max_leverage=2):
    spy_ret = pd.Series(np.diff(np.log(data.spy)), index=data.spy.index[1:])
    excess_ret = np.power(1 + spy_ret, 365.25) - two_year
    ex_stdev = data.vix / 100
    ex_var = np.power(ex_stdev, 2)
    f = c / ex_var * excess_ret.shift(1)
    cond_ret = f + two_year
    risk_ret_trade = np.minimum(cond_ret / ex_var, max_leverage)
    risk_free_weight = np.maximum(1 - risk_ret_trade, 0)
    weights = {"spy": risk_ret_trade, "vgsh": risk_free_weight}
    return bt.Strategy(
        "vol_managed_vix",
        [
            bt.algos.SelectThese(["spy", "vgsh"]),
            bt.algos.RunDaily(),
            TSDWeights(**weights),
            bt.algos.Rebalance(),
        ],
    )


def vol_managed_potfolio(data, c, max_leverage=2):
    spy_ret = pd.Series(np.diff(np.log(data.spy)), index=data.spy.index[1:])
    excess_ret = np.power(1 + spy_ret, 365.25) - two_year
    realized_std = spy_ret.rolling(21).std() * math.sqrt(
        253
    )  # get the rolling standard deviations
    realized_var = np.power(realized_std, 2)  # square it to come up with the variance
    f = (
        c / realized_var * excess_ret.shift(1)
    )  # formula from https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
    cond_ret = f + two_year

    # in your paper you mention the optimal weight being proportional to the risk return trade off
    # this attempts to capture that.
    risk_ret_trade = np.minimum(
        cond_ret / realized_var, max_leverage
    )  # this strategy seems to go bankrupt if we let it get too crazy with the leverage

    # we put the rest of the allocation into some risk-free investment
    # but we don't short the bond etf
    risk_free_weight = np.maximum(1 - risk_ret_trade, 0)

    weights = {
        "spy": risk_ret_trade,
        "vgsh": risk_free_weight,
    }  # a map of weights that we allocate to each ticker overtime
    return bt.Strategy(
        "vol_managed",
        [
            bt.algos.SelectThese(["spy", "vgsh"]),
            bt.algos.RunAfterDays(42),
            bt.algos.RunDaily(),
            TSDWeights(**weights),
            bt.algos.Rebalance(),
        ],
    )


def eighty_twenty():

    return bt.Strategy(
        "80_20",
        [
            bt.algos.SelectThese(["spy", "vgsh"]),
            bt.algos.RunMonthly(),
            bt.algos.WeighSpecified(spy=0.8, vgsh=0.2),
            bt.algos.Rebalance(),
        ],
    )


def fictional_vix():

    return bt.Strategy(
        "vix_buys",
        [
            bt.algos.SelectThese(["spy", "vix"]),
            bt.algos.RunMonthly(),
            bt.algos.WeighSpecified(spy=0.9, vix=0.1),
            bt.algos.Rebalance(),
        ],
    )


def buy_and_hold():

    return bt.Strategy(
        "buy_and_hold",
        [
            bt.algos.SelectThese(["spy"]),
            bt.algos.RunMonthly(),
            bt.algos.WeighSpecified(spy=1),
            bt.algos.Rebalance(),
        ],
    )


data = bt.get("^vix, spy, vgsh", start="2017-01-01", end="2022-03-11")


def run_backtests():
    tests = [
        bt.Backtest(vol_managed_potfolio_vix(data, 0.00426746700832415), data),
        bt.Backtest(vol_managed_potfolio(data, 0.00426746700832415), data),
        bt.Backtest(eighty_twenty(), data),
        bt.Backtest(buy_and_hold(), data),
        bt.Backtest(fictional_vix(), data),
    ]

    res = bt.run(*tests)
    res.display()

    plot = res.plot()
    plot.figure.show()


run_backtests()
