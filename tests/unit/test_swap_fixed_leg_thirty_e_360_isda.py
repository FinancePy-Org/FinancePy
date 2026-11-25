"""30E/360 (ISDA), ISDA 2006 Definitions section 4.16(h): D2 is changed to 30
when it is the last day of February, unless it is the Termination Date."""

import pytest

from financepy.products.rates.swap_fixed_leg import SwapFixedLeg
from financepy.utils.calendar import BusDayAdjustTypes, CalendarTypes
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.global_types import SwapTypes

NOTIONAL = 10_000_000
COUPON = 0.04

########################################################################################


def _fixed_leg(effective_dt, termination_dt, freq_type, end_of_month=False):
    return SwapFixedLeg(
        effective_dt,
        termination_dt,
        SwapTypes.RECEIVE,
        COUPON,
        freq_type,
        DayCountTypes.THIRTY_E_360_ISDA,
        notional=NOTIONAL,
        cal_type=CalendarTypes.WEEKEND,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        end_of_month=end_of_month,
    )


########################################################################################


def test_final_period_ending_28_february_is_not_rolled_to_30():

    leg = _fixed_leg(Date(28, 2, 2025), Date(28, 2, 2030), FrequencyTypes.ANNUAL)

    # 28 Feb 2028 to 28 Feb 2029 ends on the last day of February but not on
    # the termination date, so D2 becomes 30: 360 + 30 - 28 = 362 days
    assert leg.end_accrued_dts[3] == Date(28, 2, 2029)
    assert leg.year_fracs[3] == pytest.approx(362 / 360, abs=1e-12)

    # 28 Feb 2029 to 28 Feb 2030 ends on the termination date, so D1 becomes
    # 30 but D2 stays 28: 360 + 28 - 30 = 358 days
    assert leg.end_accrued_dts[-1] == Date(28, 2, 2030)
    assert leg.accrued_days[-1] == 358
    assert leg.year_fracs[-1] == pytest.approx(358 / 360, abs=1e-12)
    assert leg.payments[-1] == pytest.approx(NOTIONAL * COUPON * 358 / 360, abs=1e-6)


########################################################################################


def test_final_period_ending_29_february_is_not_rolled_to_30():

    leg = _fixed_leg(
        Date(29, 2, 2024),
        Date(29, 2, 2028),
        FrequencyTypes.SEMI_ANNUAL,
        end_of_month=True,
    )

    # 31 Aug 2027 to the termination date 29 Feb 2028: D1 = 30 and D2 = 29,
    # so 30 * (2 - 8) + 360 + 29 - 30 = 179 days
    assert leg.start_accrued_dts[-1] == Date(31, 8, 2027)
    assert leg.end_accrued_dts[-1] == Date(29, 2, 2028)
    assert leg.accrued_days[-1] == 179
    assert leg.year_fracs[-1] == pytest.approx(179 / 360, abs=1e-12)


########################################################################################
