import datetime
from dateutil.relativedelta import relativedelta
import ffn
import numpy as np

from backtest import risk_return_trade

C = 0.02576024679
RF = 2.247 / 100
MAX_LEVERAGE = 3

today = datetime.datetime.today()
one_month_ago = today - relativedelta(months=1)
yesterday = today - relativedelta(days=1)
prices = ffn.get("spy", start=one_month_ago, end=yesterday)

log_ret = np.log(prices.spy / prices.spy.shift(1))
l = risk_return_trade(C, log_ret.var() * 252, 0.105, MAX_LEVERAGE, RF)
risk_free_weight = np.maximum(1 - l, 0)
sso_weight = np.maximum((l - 1) / 2, 0)
spy_weight = 1 - sso_weight - risk_free_weight

print("Expected Equity Exposure: %f\n" % l)
print("SPY: %f" % spy_weight)
print("SSO: %f" % sso_weight)
print("Risk Free: %f" % risk_free_weight)
