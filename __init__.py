import datetime

import ffn
import numpy as np

from backtest import risk_return_trade

rf = 1.31 / 100
max_leverage = 3

today = datetime.datetime.today()
one_month_ago = today.replace(month=today.month - 1)
prices = ffn.get('spy', start=one_month_ago)

log_ret = np.diff(np.log(prices.spy))
l = risk_return_trade(0.03380993, log_ret.var() * 252, 0.105, max_leverage, rf)
risk_free_weight = np.maximum(1 - l, 0)
sso_weight = np.maximum((l - 1) / 2, 0)
spy_weight = 1 - sso_weight - risk_free_weight

print("Expected Equity Exposure: %f\n" % l)
print("SPY: %f" % spy_weight)
print("SSO: %f" % sso_weight)
print("Risk Free: %f" % risk_free_weight)
