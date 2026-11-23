from decimal import Decimal, localcontext

import pytest

from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.products.equity.equity_variance_swap import EquityVarianceSwap
from financepy.products.fx.fx_variance_swap import FinFXVarianceSwap
from financepy.utils.date import Date


@pytest.mark.parametrize("swap_type", [EquityVarianceSwap, FinFXVarianceSwap])
@pytest.mark.parametrize("pay_strike", [None, True, False], ids=["default", "pay", "receive"])
@pytest.mark.parametrize(
    "elapsed, realised, remaining",
    [
        (0, "0.25", "0.09"),
        (91, "0.09", "0.01"),
        (200, "0.08", "0.01"),
        (365, "0.01", "0.25"),
        (365, "0.04", "0.25"),
    ],
    ids=["inception", "mid_below", "mid_above", "maturity", "par"],
)
def test_variance_swap_direction(swap_type, pay_strike, elapsed, realised, remaining):
    """Receiving the fixed strike reverses the full discounted net payoff."""
    start = Date(1, 1, 2025)
    maturity = Date(1, 1, 2026)
    value_dt = start.add_days(elapsed)
    curve = FlatDiscountCurve(value_dt, 0.031)  # continuous ACT/365F
    kwargs = {} if pay_strike is None else {"pay_strike_flag": pay_strike}
    swap = swap_type(start, maturity, 0.04, notional=2_000_000.0, **kwargs)

    # Exact 365-day payoff identity, written without FinancePy day count or
    # discount routines. Calendar days are known from the fixture.
    with localcontext() as ctx:
        ctx.prec = 60
        total = Decimal(365)
        elapsed = Decimal(elapsed)
        left = total - elapsed
        df = (-Decimal("0.031") * left / total).exp()
        received_var = (elapsed * Decimal(realised) + left * Decimal(remaining)) / total
        signed = -1 if pay_strike is False else 1
        expected = signed * df * Decimal(2_000_000) * (received_var - Decimal("0.04"))

    actual = swap.value(value_dt, float(realised), float(remaining), curve)
    # The library flat curve floors exact zero time at 1e-12, so its
    # same-day discount can differ from one by a few dozen ULPs.
    assert actual == pytest.approx(float(expected), rel=1e-12, abs=1e-10)
