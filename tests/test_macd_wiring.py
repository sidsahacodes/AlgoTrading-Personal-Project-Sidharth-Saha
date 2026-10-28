import numpy as np
import pandas as pd
import pandas_ta as ta


def test_explicit_macd_params_equal_library_defaults():
    """Wiring MACD_* from config must not change the baseline output."""
    from momentum_rebalancer.config import MACD_FAST, MACD_SIGNAL, MACD_SLOW
    px = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(size=300)))
    default  = ta.macd(px).iloc[:, 0]
    explicit = ta.macd(px, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL).iloc[:, 0]
    pd.testing.assert_series_equal(default, explicit)
