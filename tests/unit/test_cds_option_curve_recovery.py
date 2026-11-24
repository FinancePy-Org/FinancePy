"""CDS and CDS index options price the forward spread with the curve recovery."""

import pytest

from financepy.market.curves.cds_curve import CDSCurve
from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.products.credit.cds import CDS
from financepy.products.credit.cds_index_option import CDSIndexOption
from financepy.products.credit.cds_option import CDSOption
from financepy.utils.date import Date

CLEAN = 1

value_dt = Date(1, 1, 2026)
expiry_dt = Date(1, 2, 2026)
maturity_dt = Date(20, 12, 2030)
libor_curve = FlatDiscountCurve(value_dt, 0.03)
notional = 1_000_000.0
market_spread = 0.01
strike = 0.009


def _flat_issuer_curve(recovery):
    # Calibrated to a 100bp CDS to the option maturity whatever the recovery
    contract = CDS(value_dt, maturity_dt, market_spread)
    return CDSCurve(value_dt, [contract], libor_curve, recovery)


########################################################################################


@pytest.mark.parametrize("recovery", [0.25, 0.30, 0.40])
def test_cds_option_forward_spread_matches_market_quote(recovery):

    issuer_curve = _flat_issuer_curve(recovery)
    payer = CDSOption(expiry_dt, maturity_dt, strike, notional, True)
    receiver = CDSOption(expiry_dt, maturity_dt, strike, notional, False)

    v_pay = payer.value(value_dt, issuer_curve, 0.40)
    v_rec = receiver.value(value_dt, issuer_curve, 0.40)
    fwd_rpv01 = payer.cds.rpv01(value_dt, issuer_curve)[CLEAN]

    # Payer minus receiver is the forward CDS value (F - K) * RPV01. On a flat
    # curve the one month forward spread is the 100bp quote the curve was
    # calibrated to, so it must not depend on the recovery assumption.
    fwd_spread = strike + (v_pay - v_rec) / (fwd_rpv01 * notional)

    assert fwd_spread == pytest.approx(market_spread, abs=0.1e-4)


########################################################################################


@pytest.mark.parametrize("recovery", [0.25, 0.30])
def test_cds_index_option_forward_spread_matches_market_quote(recovery):

    option = CDSIndexOption(expiry_dt, maturity_dt, market_spread, strike, notional)
    sigma = 0.40

    # With the 100bp quote held fixed, the recovery assumption only has a second
    # order effect on the option, so compare with a 40% recovery curve.
    base_curve = _flat_issuer_curve(0.40)
    issuer_curve = _flat_issuer_curve(recovery)

    base_black = option.value_adjusted_black(value_dt, base_curve, 0.40, libor_curve, sigma)
    black = option.value_adjusted_black(value_dt, issuer_curve, recovery, libor_curve, sigma)

    assert black[0] == pytest.approx(base_black[0], rel=0.01)
    assert black[1] == pytest.approx(base_black[1], rel=0.03)

    # The Anderson forward spread x is in bp and is the quote plus the
    # front-end protection, which is close to 102bp for either recovery
    base_x = option.value_anderson(value_dt, [base_curve] * 5, 0.40, sigma)[3]
    x = option.value_anderson(value_dt, [issuer_curve] * 5, recovery, sigma)[3]

    assert x == pytest.approx(base_x, abs=0.1)
