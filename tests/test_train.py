import numpy as np
from neural_flow import Flow, train
from neural_flow.bijectors import rolling_spline_coupling
import pytest


@pytest.mark.filterwarnings("error:RuntimeWarning")
def test_bad_input_distribution():
    rng = np.random.default_rng(1)
    x = rng.pareto(5, size=1000)
    flow = Flow(rolling_spline_coupling(2))
    X = np.column_stack((x, x))
    # this should not raise RuntimeWarning
    loss_train = train(flow, X, X)[1]
    assert np.all(np.isfinite(loss_train))
