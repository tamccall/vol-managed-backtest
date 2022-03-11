import bt
import numpy as np


class TSDWeights(bt.algos.Algo):
    def __call__(self, target):
        date = target.now
        tmp_weight = {}
        for k, v in self.weights.items():
            tmp_weight[k] = v.at[date]
        target.temp["weights"] = tmp_weight
        return True

    def __init__(self, **weights):
        super().__init__("TSDWeights")
        self.weights = weights


def vol_managed_potfolio(data, c, rf, expected_ret):
    ex_stdev = data["vix"] / 100
    ex_var = np.power(ex_stdev, 2)
    f = c / ex_var * (expected_ret - rf)
    cond_ret = f + rf
    risk_ret_trade = cond_ret / ex_var
    risk_free_weight = 1 - risk_ret_trade
    risk_free_weight = np.maximum(risk_free_weight, 0)
    weights = {"spy": risk_ret_trade, "vgsh": risk_free_weight}
    return bt.Strategy(
        "vol_managed",
        [
            bt.algos.SelectThese(["spy", "vgsh"]),
            bt.algos.RunMonthly(),
            TSDWeights(**weights),
            bt.algos.Rebalance(),
        ],
    )


def benchmark():

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

data = bt.get("^vix, spy, vgsh", start="2000-01-01")

tests = [
    bt.Backtest(vol_managed_potfolio(data, 0.00426746700832415, 0.0185, 0.1), data),
    bt.Backtest(benchmark(), data),
    bt.Backtest(fictional_vix(), data)
]

res = bt.run(*tests)
res.display()

plot = res.plot()
plot.figure.show()
