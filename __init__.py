import datetime
from dateutil.relativedelta import relativedelta
import ffn
import numpy as np

from backtest import risk_return_trade

C = 0.03380993
RF = 1.31 / 100
MAX_LEVERAGE = 3

today = datetime.datetime.today()
one_month_ago = today - relativedelta(month=1)
yesterday = today - relativedelta(day=1)
prices = ffn.get('spy', start=one_month_ago, end=yesterday)

log_ret = np.log(1 + prices.spy.pct_change())
l = risk_return_trade(C, log_ret.var() * 252, 0.105, MAX_LEVERAGE, RF)
risk_free_weight = np.maximum(1 - l, 0)
sso_weight = np.maximum((l - 1) / 2, 0)
spy_weight = 1 - sso_weight - risk_free_weight

print("Expected Equity Exposure: %f\n" % l)
print("SPY: %f" % spy_weight)
print("SSO: %f" % sso_weight)
print("Risk Free: %f" % risk_free_weight)
