# Copyright (C) 2018, 2019, 2020 Dominic O'Kane

import numpy as np
import pytest

from financepy.utils.date import Date
from financepy.utils.error import FinError
from financepy.models.black_scholes import BlackScholes
from financepy.models.bs_american_analytic import BlackScholesAlgorithmTypes
from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.utils.global_types import OptionTypes
from financepy.products.equity.equity_american_option import (
    EquityAmericanOption,
)


value_dt = Date(1, 1, 2016)
expiry_dt = Date(1, 1, 2017)
stock_price = 50.0
interest_rate = 0.06
dividend_yield = 0.04
volatility = 0.40
strike_price = 50.0
num_steps = 100

discount_curve = FlatDiscountCurve(value_dt, interest_rate)
dividend_curve = FlatDiscountCurve(value_dt, dividend_yield)

model = BlackScholes(volatility)


@pytest.mark.parametrize("spot", [50, 50.0, np.int64(50), np.float64(50), np.array(50.0)])
def test_scalar_spot_representations_return_one_price(spot):
    """Python/NumPy scalar representations have the same contract value."""
    option = EquityAmericanOption(expiry_dt, strike_price, OptionTypes.AMERICAN_PUT, 3.0)
    value = option.value(value_dt, spot, discount_curve, dividend_curve, model)
    assert np.ndim(value) == 0
    assert round(value / 3.0, 3) == 7.255


########################################################################################


def test_european_put():

    put_option = EquityAmericanOption(
        expiry_dt, strike_price, OptionTypes.EUROPEAN_PUT
    )

    value = put_option.value(
        value_dt, stock_price, discount_curve, dividend_curve, model
    )

    assert round(value, 3) == 7.081


########################################################################################


def test_american_put():

    put_option = EquityAmericanOption(
        expiry_dt, strike_price, OptionTypes.AMERICAN_PUT
    )

    value = put_option.value(
        value_dt, stock_price, discount_curve, dividend_curve, model
    )

    assert round(value, 3) == 7.255


########################################################################################


def test_european_call():

    call_option = EquityAmericanOption(
        expiry_dt, strike_price, OptionTypes.EUROPEAN_CALL
    )

    value = call_option.value(
        value_dt, stock_price, discount_curve, dividend_curve, model
    )

    assert round(value, 3) == 8.035


########################################################################################


def test_american_call():

    call_option = EquityAmericanOption(
        expiry_dt, strike_price, OptionTypes.AMERICAN_CALL
    )

    value = call_option.value(
        value_dt, stock_price, discount_curve, dividend_curve, model
    )

    assert round(value, 3) == 8.056


########################################################################################


@pytest.mark.parametrize(
    "opt_type, expected",
    [
        (OptionTypes.AMERICAN_CALL, 8.056),
        (OptionTypes.AMERICAN_PUT, 7.254),
    ],
)
def test_american_leisen_reimer(opt_type, expected):
    """The Leisen-Reimer tree can be selected for American exercise."""

    option = EquityAmericanOption(expiry_dt, strike_price, opt_type)

    value = option.value(
        value_dt,
        stock_price,
        discount_curve,
        dividend_curve,
        model,
        algo_type=BlackScholesAlgorithmTypes.LEISEN_REIMER,
    )

    assert round(value, 3) == expected


########################################################################################


@pytest.mark.parametrize(
    "algo_type",
    [BlackScholesAlgorithmTypes.DEFAULT, BlackScholesAlgorithmTypes.FINITE_DIFFERENCE],
)
def test_negative_dividend_yield_is_rejected(algo_type):
    """A stock's dividend yield cannot be negative. The pricers accept a negative
    q because it is the foreign rate for FX options, so the equity product checks it."""

    option = EquityAmericanOption(expiry_dt, strike_price, OptionTypes.AMERICAN_PUT)

    with pytest.raises(FinError, match="Dividend yield"):
        option.value(
            value_dt,
            stock_price,
            discount_curve,
            FlatDiscountCurve(value_dt, -0.01),
            model,
            algo_type=algo_type,
        )


########################################################################################


def test_negative_interest_rate_is_allowed():
    """Interest rates can be negative for equity options too."""

    option = EquityAmericanOption(expiry_dt, strike_price, OptionTypes.AMERICAN_PUT)

    value = option.value(
        value_dt,
        stock_price,
        FlatDiscountCurve(value_dt, -0.01),
        dividend_curve,
        model,
        algo_type=BlackScholesAlgorithmTypes.FINITE_DIFFERENCE,
    )

    assert value > 0.0
