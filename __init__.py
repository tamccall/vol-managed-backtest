import math
import random

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
two_year_daily = np.power(1 + two_year, 1 / 365) - 1


def vol_managed_potfolio_vix(data, c, expected_ret, max_leverage=2):
    risk_ret_trade = risk_return_trade_from_vix(data, c, expected_ret, max_leverage)
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
    risk_ret_trade = risk_return_tradeoff_from_spy(data, c, expected_ret, max_leverage)

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
            bt.algos.RunMonthly(),
            TSDWeights(**weights),
            bt.algos.Rebalance(),
        ],
    )

def risk_return_trade(c, ex_var, expected_ret, max_leverage):
    excess_ret = expected_ret - two_year
    # formula from https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
    f = c / ex_var * excess_ret
    # get the conditional expected return
    cond_ret = f + two_year

    # put some bounds on it to constrain the leverage
    risk_ret_trade = np.minimum(cond_ret / ex_var, max_leverage)
    # but don't short it
    risk_ret_trade = np.maximum(risk_ret_trade, 0)
    return risk_ret_trade


def risk_return_trade_from_vix(data, c, expected_ret, max_leverage):
    # convert the vix into a variation
    ex_stdev = data.vix / 100 # convert the vix into an annulized std
    ex_var = np.power(ex_stdev, 2) # square it to get the variation
    yest_vix = ex_var.shift(1) # make sure we are working with yesterday's vix

    return risk_return_trade(c, yest_vix, expected_ret, max_leverage)


def risk_return_tradeoff_from_spy(data, c,expected_ret, max_leverage=2):
    spy_ret = pd.Series(np.diff(np.log(data.spy)), index=data.spy.index[1:])

    # get the rolling standard deviations
    realized_std = spy_ret.rolling(21).std() * math.sqrt(252)

    # square it to come up with the variance
    realized_var = np.power(realized_std, 2).shift(1)

    return risk_return_trade(c, realized_var, expected_ret, max_leverage)


def vol_managed_potfolio_etf(data, c, expected_ret, max_leverage=2):
    risk_ret_trade = risk_return_trade_from_vix(data, c, expected_ret, max_leverage)

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


data = bt.get("^vix, spy, vgsh, sso", start="2000-04-17", end="2022-03-11")

# code from https://twitter.com/alan_econ/status/1502372674009538564?s=20&t=McRHU8l5Cpbtg-MHa8BoZQ
def get_c():
    spy_ret = pd.Series(np.diff(np.log(data.spy)), index=data.spy.index[1:])
    mkt_rf = spy_ret - two_year_daily

    var = np.power(mkt_rf.rolling(21).std() * math.sqrt(252), 2)
    smkt = mkt_rf / var.shift(1)
    return math.sqrt(mkt_rf.var() / smkt.var()) # gets us a c that will ensure the same variance


def run_backtests():
    tests = [
        bt.Backtest(
            vol_managed_potfolio(data, 0.04198355, 0.105, max_leverage=2), data
        ),
        bt.Backtest(
            vol_managed_potfolio_vix(data, 0.04198355, 0.105, max_leverage=2),
            data,
        ),
        bt.Backtest(
            vol_managed_potfolio_etf(data, 0.04198355, 0.105, max_leverage=2),
            data,
        ),
        bt.Backtest(eighty_twenty(), data),
        bt.Backtest(buy_and_hold(), data),
    ]

    res = bt.run(*tests)
    res.display()
    plot = res.plot()
    plot.figure.show()

    mkt_excess = res['buy_and_hold'].log_returns - two_year_daily
    vol_managed_excess = res['vol_managed_vix'].log_returns - two_year_daily

    reg = scipy.stats.linregress(mkt_excess.dropna(), vol_managed_excess.dropna())
    print("\nBuy and hold regression:")
    print(reg)

    print("\nVol managed ETF regression:")
    etf_excess = res['vol_managed_etf'].log_returns - two_year_daily
    reg = scipy.stats.linregress(vol_managed_excess.dropna(), etf_excess.dropna())
    print(reg)

    print("\nVol managed ETF regression (mkt):")
    reg = scipy.stats.linregress(mkt_excess.dropna(), etf_excess.dropna())
    print(reg)


def backtest_for_c_l(arr):
    s = data.head(int(len(data) / 2))
    c, l = arr

    res = bt.run(
        bt.Backtest(vol_managed_potfolio(s, c, 0.1, max_leverage=l), s),
        bt.Backtest(buy_and_hold(), s)
    )

    excess_bh = (res['buy_and_hold'].log_returns - two_year_daily).dropna()
    excess_vol = (res['vol_managed'].log_returns - two_year_daily).dropna()
    reg = scipy.stats.linregress(excess_bh, excess_vol)
    return -1 * reg.intercept


def optimize_c():
    res = scipy.optimize.minimize(
        backtest_for_c_l,
        x0=[get_c(), random.uniform(1.0, 2.0)],
        bounds=[(0.00001, 100.0), (1, 2.0)],
        method="Nelder-Mead",
    )

    print(res)

run_backtests()