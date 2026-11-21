"""Regression for optional repricing of OIS calibration instruments."""

import pytest

from financepy.market.curves.ois_curve import OISCurve
from financepy.products.rates.ois_curve import OISCurve as LegacyOISCurve
from financepy.products.rates.ois import OIS
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.error import FinError
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.global_types import SwapTypes


@pytest.mark.parametrize("curve_class", (OISCurve, LegacyOISCurve))
def test_ois_curve_reprices_with_optional_check(curve_class):
    anchor = Date(3, 1, 2024)
    swaps = [
        OIS(anchor, tenor, SwapTypes.PAY, quote, FrequencyTypes.ANNUAL,
            DayCountTypes.ACT_360, notional=10.0)
        for tenor, quote in (("1Y", 0.025), ("2Y", 0.028))
    ]
    curve = curve_class(anchor, [], [], swaps, check_refit_flag=True)
    for swap in swaps:
        assert abs(swap.value(anchor, curve)) / swap.fixed_leg.notional < 1e-8


def test_ois_curve_check_still_rejects_off_market_swap():
    anchor = Date(3, 1, 2024)
    dc = DayCountTypes.ACT_360
    swap = OIS(anchor, "1Y", SwapTypes.PAY, 0.025, FrequencyTypes.ANNUAL, dc)
    curve = OISCurve(anchor, [], [], [swap])
    curve.used_swaps[0] = OIS(anchor, "1Y", SwapTypes.PAY, 0.04, FrequencyTypes.ANNUAL, dc)
    with pytest.raises(FinError, match="Swap not repriced"):
        curve.check_refit(1e-5, 1e-5)
