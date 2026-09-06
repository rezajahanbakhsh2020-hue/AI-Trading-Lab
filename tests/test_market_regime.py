import pandas as pd
import pytest
from src.evaluation.market_regime import classify_volatility_regime

def df():
    return pd.DataFrame({"return":[.01,.02,-.01,.015,-.005,.012,-.008,.011,.009,-.006]})

def test_columns():
    r=classify_volatility_regime(df(),window=3)
    assert {"volatility","volatility_median","volatility_ratio","volatility_regime"}<=set(r)

def test_no_modify():
    x=df(); old=x.copy()
    classify_volatility_regime(x,window=3)
    pd.testing.assert_frame_equal(x,old)

def test_labels():
    r=classify_volatility_regime(df(),window=3)
    assert set(r.volatility_regime.dropna())<={
        "low_volatility","normal_volatility","high_volatility"}

def test_missing():
    with pytest.raises(ValueError):
        classify_volatility_regime(pd.DataFrame({"close":[1,2,3]}))

def test_window():
    with pytest.raises(ValueError):
        classify_volatility_regime(df(),window=0)
