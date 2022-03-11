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


def vol_managed_potfolio_vix(data, c, rf, expected_ret):
    ex_stdev = data.vix / 100
    ex_var = np.power(ex_stdev, 2)
    f = c / ex_var * (expected_ret - rf)
    cond_ret = f + rf
    risk_ret_trade = cond_ret / ex_var
    risk_free_weight = 1 - risk_ret_trade
    risk_free_weight = np.maximum(risk_free_weight, 0)
    weights = {"spy": risk_ret_trade, "vgsh": risk_free_weight}
    return bt.Strategy(
        "vol_managed_vix",
        [
            bt.algos.SelectThese(["spy", "vgsh"]),
            bt.algos.RunMonthly(),
            TSDWeights(**weights),
            bt.algos.Rebalance(),
        ],
    )

def vol_managed_potfolio(data, c, rf, expected_ret):
    spy_ret = pd.Series(np.diff(np.log(data.spy)), index=data.spy.index[1:])
    ex_std = spy_ret.rolling(21).std() * math.sqrt(253) # get the rolling standard deviation
    ex_var = np.power(ex_std, 2) # square it to come up with the variance
    f = c / ex_var * (expected_ret - rf) # formula from https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
    cond_ret = f + rf

    # in your paper you mention the optimal weight being proportional to the risk return trade off
    # this attempts to capture that.
    risk_ret_trade = cond_ret / ex_var

    risk_ret_trade = np.minimum(risk_ret_trade.dropna(), 2) # this strategy seems to go bankrupt if we let it get too crazy with the leverage

    # we put the rest of the allocation into some risk-free investment
    risk_free_weight = 1 - risk_ret_trade

    # but we don't short the bond etf
    risk_free_weight = np.maximum(risk_free_weight, 0)


    weights = {"spy": risk_ret_trade, "vgsh": risk_free_weight} # a map of weights that we allocate to each ticker overtime
    return bt.Strategy(
        "vol_managed",
        [
            bt.algos.SelectThese(["spy", "vgsh"]),
            bt.algos.RunAfterDays(30),
            bt.algos.RunMonthly(),
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

data = bt.get("^vix, spy, vgsh", start="2012-01-01")

tests = [
    bt.Backtest(vol_managed_potfolio_vix(data, 0.00426746700832415, 0.0185, 0.1), data),
    bt.Backtest(vol_managed_potfolio(data, 0.00426746700832415, 0.0185, 0.1), data),
    bt.Backtest(eighty_twenty(), data),
    bt.Backtest(buy_and_hold(), data),
    bt.Backtest(fictional_vix(), data)
]

res = bt.run(*tests)
res.display()

plot = res.plot()
plot.figure.show()
