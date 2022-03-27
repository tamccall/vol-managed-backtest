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
            try:
                weight = v.at[date]
                if math.isnan(weight):
                    return False
                else:
                    tmp_weight[k] = weight
            except KeyError:
                return False

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


def vol_managed_potfolio(
    data, c, expected_ret, max_leverage=2, strategy_name="vol_managed"
):
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
        strategy_name,
        [
            bt.algos.SelectThese(["spy", "vgsh"]),
            bt.algos.RunAfterDays(42),
            bt.algos.RunMonthly(),
            TSDWeights(**weights),
            bt.algos.Rebalance(),
        ],
    )


def risk_return_trade(c, ex_var, expected_ret, max_leverage, rf):
    excess_ret = expected_ret - rf
    # formula from https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
    f = (c / ex_var) * excess_ret
    # put some bounds on it to constrain the leverage
    lev = f / ex_var
    risk_ret_trade = np.minimum(lev, max_leverage)
    # but don't short it
    risk_ret_trade = np.maximum(risk_ret_trade, 0)
    return risk_ret_trade


def risk_return_trade_from_vix(data, c, expected_ret, max_leverage):
    # convert the vix into a variation
    ex_stdev = data.vix / 100  # convert the vix into an annulized std
    ex_var = np.power(ex_stdev, 2)  # square it to get the variation
    yest_vix = ex_var.shift(1)  # make sure we are working with yesterday's vix

    return risk_return_trade(c, yest_vix, expected_ret, max_leverage, two_year)


def risk_return_tradeoff_from_spy(data, c, expected_ret, max_leverage=2):
    spy_ret = np.log(data.spy / data.spy.shift(1))

    # get the rolling standard deviations and annualize it
    realized_var = spy_ret.rolling(21).var() * 252

    return risk_return_trade(
        c, realized_var.shift(1), expected_ret, max_leverage, two_year
    )


def vol_managed_potfolio_etf(data, c, expected_ret, max_leverage=2):
    risk_ret_trade = risk_return_tradeoff_from_spy(data, c, expected_ret, max_leverage)

    # we put the rest of the allocation into some risk-free investment
    # but we don't short the bond etf
    risk_free_weight = np.maximum(1 - risk_ret_trade, 0)
    sso_weight = np.maximum((risk_ret_trade - 1) / 2, 0)
    spy_weight = 1 - sso_weight - risk_free_weight

    weights = {
        "spy": spy_weight,
        "upro": sso_weight,
        "vgsh": risk_free_weight,
    }  # a map of weights that we allocate to each ticker overtime
    return bt.Strategy(
        "vol_managed_etf",
        [
            bt.algos.SelectThese(["spy", "vgsh", "upro"]),
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


data = bt.get("^vix, spy, vgsh, upro", start="2000-04-17", end="2022-03-11")

# code from https://twitter.com/alan_econ/status/1502372674009538564?s=20&t=McRHU8l5Cpbtg-MHa8BoZQ
def get_c():
    spy_ret = np.log(data.spy / data.spy.shift(1))
    mkt_rf = spy_ret - two_year_daily

    var = np.power(mkt_rf.rolling(21).std(), 2)
    smkt = mkt_rf / var.shift(1)
    return math.sqrt(
        mkt_rf.var() / smkt.var()
    ) * 252


def med_vix_c():
    med_vix = data.vix.median() / 100
    vix_var = math.pow(med_vix, 2)

    # corresponds to an 80:20 allocation on average
    return vix_var


def run_backtests():
    tests = [
        bt.Backtest(
            vol_managed_potfolio(data, 0.03101121352, 0.105, max_leverage=3), data
        ),
        bt.Backtest(
            vol_managed_potfolio_etf(data, 0.03101121352, 0.105, max_leverage=3),
            data,
        ),
        bt.Backtest(eighty_twenty(), data),
        bt.Backtest(buy_and_hold(), data),
    ]

    res = bt.run(*tests)
    res.display()

    plot_weights(res)
    display_returns(res)
    print_regressions(res)


def plot_weights(res):
    plot = (
        res.backtests["vol_managed"]
        .weights.drop("vol_managed", axis=1)
        .plot(figsize=(10, 7))
    )
    plot.figure.show()
    plot = (
        res.backtests["vol_managed_etf"]
        .weights.drop("vol_managed_etf", axis=1)
        .plot(figsize=(10, 7))
    )
    plot.figure.show()


def display_returns(res):
    plot = res.plot(figsize=(10, 7))
    plot.figure.show()


def print_regressions(res):
    mkt_excess = res["buy_and_hold"].log_returns - two_year_daily
    vol_managed_excess = res["vol_managed"].log_returns - two_year_daily
    reg = scipy.stats.linregress(mkt_excess.dropna(), vol_managed_excess.dropna())
    print("\nBuy and hold regression:")
    print(reg)
    print("\nVol managed ETF regression:")
    etf_excess = res["vol_managed_etf"].log_returns - two_year_daily
    reg = scipy.stats.linregress(vol_managed_excess.dropna(), etf_excess.dropna())
    print(reg)
    print("\nVol managed ETF regression (mkt):")
    reg = scipy.stats.linregress(mkt_excess.dropna(), etf_excess.dropna())
    print(reg)


def backtest_for_c_l(arr):
    c, l = arr

    res = bt.run(
        bt.Backtest(vol_managed_potfolio(data, c, 0.105, max_leverage=l), data),
        bt.Backtest(buy_and_hold(), data),
    )

    excess_bh = (res["buy_and_hold"].log_returns - two_year_daily).dropna()
    excess_vol = (res["vol_managed"].log_returns - two_year_daily).dropna()
    reg = scipy.stats.linregress(excess_bh, excess_vol)
    return -1 * reg.intercept


def optimize_c():
    res = scipy.optimize.minimize(
        backtest_for_c_l,
        x0=[med_vix_c(), random.uniform(1.0, 2.0)],
        bounds=[(0.00001, 100.0), (1, 3.0)],
        method="Nelder-Mead",
    )

    print(res)
