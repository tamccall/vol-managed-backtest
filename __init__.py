import bt
import numpy as np


def vol_managed_potfolio(data, c, rf, expected_ret):
    ex_stdev = data["vix"] / 100
    ex_var = np.power(ex_stdev, 2)
    f = c / ex_var * (expected_ret - rf)
    cond_ret = f + rf
    risk_ret_trade = cond_ret / ex_var
    risk_ret_trade = np.minimum(risk_ret_trade, 1)
    risk_free_weight = 1 - risk_ret_trade
    weights = {"spy": risk_ret_trade, "vgsh": risk_free_weight}
    return bt.Strategy(
        "vol_managed",
        [
            bt.algos.SelectThese(["spy", "vgsh"]),
            bt.algos.RunMonthly(),
            bt.algos.WeighSpecified(**weights),
        ],
    )


def benchmark(data):
    weights = {
        "spy": np.full((len(data.spy.values), 1), 0.8),
        "vgsh": np.full((len(data.vgsh.values), 1), 0.2),
    }
    return bt.Strategy(
        "vol_managed",
        [
            bt.algos.SelectThese(["spy", "vgsh"]),
            bt.algos.RunMonthly(),
            bt.algos.WeighSpecified(**weights),
        ],
    )


data = bt.get("^vix, spy, vgsh", start="2010-01-01")
# strat = vol_managed_potfolio(data, 0.00426746700832415, 0.0185, 0.1)
strat = benchmark(data)
t = bt.Backtest(strat, data, progress_bar=True)
res = bt.run(t)
res.display()
