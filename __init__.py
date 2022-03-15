import math

import bt
import numpy as np
import pandas as pd
import scipy.optimize


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


def vol_managed_potfolio_vix(data, c, expected_ret, max_leverage=2):
    excess_ret = expected_ret - two_year
    ex_stdev = data.vix / 100
    ex_var = np.power(ex_stdev, 2)
    f = c / ex_var * excess_ret
    cond_ret = f + two_year
    risk_ret_trade = np.minimum(cond_ret / ex_var, max_leverage)
    risk_ret_trade = np.maximum(risk_ret_trade, 0)
    risk_free_weight = np.maximum(1 - risk_ret_trade, 0)
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


def vol_managed_potfolio(data, c, expected_ret, max_leverage=2):
    # in your paper you mention the optimal weight being proportional to the risk return trade off
    # this attempts to capture that.
    risk_ret_trade = np.minimum(
        risk_return_tradeoff(c, data, expected_ret), max_leverage
    )  # this strategy seems to go bankrupt if we let it get too crazy with the leverage
    risk_ret_trade =np.maximum(risk_ret_trade, 0) # dont short it
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
            bt.algos.RunWeekly(),
            TSDWeights(**weights),
            bt.algos.Rebalance(),
        ],
    )


def risk_return_tradeoff(c, data, expected_ret):
    spy_ret = pd.Series(np.diff(np.log(data.spy)), index=data.spy.index[1:])
    excess_ret = expected_ret - two_year
    realized_std = spy_ret.rolling(21).std().shift(1) * math.sqrt(
        253
    )  # get the rolling standard deviations
    realized_var = np.power(realized_std, 2)  # square it to come up with the variance
    f = (
            c / realized_var * excess_ret
    )  # formula from https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
    cond_ret = f + two_year
    risk_ret_trade = cond_ret / realized_var
    return risk_ret_trade


def vol_managed_potfolio_etf(data, c, expected_ret, max_leverage=2):
    # in your paper you mention the optimal weight being proportional to the risk return trade off
    # this attempts to capture that.
    risk_ret_trade = np.minimum(
       risk_return_tradeoff(c, data, expected_ret), max_leverage
    )  # this strategy seems to go bankrupt if we let it get too crazy with the leverage
    risk_ret_trade =np.maximum(risk_ret_trade, 0) # dont short it

    # we put the rest of the allocation into some risk-free investment
    # but we don't short the bond etf
    risk_free_weight = np.maximum(1 - risk_ret_trade, 0)
    sso_weight = np.maximum(risk_ret_trade - 1, 0)
    spy_weight = 1 - sso_weight - risk_free_weight

    weights = {
        "spy": spy_weight,
        "sso": sso_weight,
        "vgsh": risk_free_weight,
    }  # a map of weights that we allocate to each ticker overtime
    return bt.Strategy(
        "vol_managed_etf",
        [
            bt.algos.SelectThese(["spy", "vgsh", "sso"]),
            bt.algos.RunAfterDays(42),
            bt.algos.RunWeekly(),
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


data = bt.get("^vix, spy, vgsh, sso", start="2000-01-01", end="2022-03-11")


def run_backtests():
    tests = [
        bt.Backtest(vol_managed_potfolio(data, 0.02341058, 0.1, max_leverage=2), data),
        # bt.Backtest(vol_managed_potfolio_etf(data, 0.03179263, 0.1, max_leverage=1.62231445), data),
        bt.Backtest(eighty_twenty(), data),
        bt.Backtest(buy_and_hold(), data),
        bt.Backtest(fictional_vix(), data),
    ]

    res = bt.run(*tests)
    res.display()
    plot = res.plot()
    plot.figure.show()

    # reg = scipy.stats.linregress(res['vol_managed'].prices, res['vol_managed_etf'].prices)
    # print(reg)



def backtest_for_c_l(arr):
    c, l = arr
    res = bt.run(bt.Backtest(vol_managed_potfolio(data, c, 0.1, max_leverage=l), data))
    return -1 * res['vol_managed'].daily_sharpe


def optimize_c():
    res = scipy.optimize.minimize(
        backtest_for_c_l,
        x0=[0.03179263, 1.62231445],
        bounds=[(0.0, 2.0), (1.0, 3.0)],
        method='Nelder-Mead'
    )

    print(res)


run_backtests()