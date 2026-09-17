"""Tests for configurable SHAP background data."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import shap  # noqa: F401

import tabicl.shap._shap as shap_module
from tabicl.shap import get_shap_explainer, get_shap_values


class DummyEstimator:
    """Minimal estimator exposing the classifier prediction API."""

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.column_stack((np.full(len(X), 0.4), np.full(len(X), 0.6)))


class CapturingExplainer:
    """Small stand-in for shap.Explainer used to inspect constructor inputs."""

    instances = []

    def __init__(self, predict_fn, background, **kwargs):
        self.predict_fn = predict_fn
        self.background = np.asarray(background)
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def __call__(self, X):
        return SimpleNamespace(feature_names=None)


@pytest.fixture(autouse=True)
def patch_explainer(monkeypatch):
    CapturingExplainer.instances.clear()
    monkeypatch.setattr(shap_module.shap, "Explainer", CapturingExplainer)


def test_get_shap_explainer_preserves_all_nan_default_background():
    X = np.zeros((4, 3))

    explainer = get_shap_explainer(DummyEstimator(), X)

    assert explainer.background.shape == (1, 3)
    assert np.isnan(explainer.background).all()


def test_get_shap_explainer_uses_custom_numeric_background():
    X = np.zeros((4, 3))
    background = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        columns=["a", "b", "c"],
    )

    explainer = get_shap_explainer(DummyEstimator(), X, X_background=background)

    np.testing.assert_array_equal(explainer.background, background.to_numpy(dtype=float))


def test_get_shap_values_forwards_background_and_preserves_feature_names():
    X_test = pd.DataFrame([[0.0, 1.0]], columns=["left", "right"])
    background = np.array([[2.0, 3.0], [4.0, 5.0]])

    explanation = get_shap_values(
        DummyEstimator(),
        X_test,
        X_background=background,
        algorithm="exact",
    )

    explainer = CapturingExplainer.instances[-1]
    np.testing.assert_array_equal(explainer.background, background)
    assert explainer.kwargs["algorithm"] == "exact"
    assert explanation.feature_names == ["left", "right"]


@pytest.mark.parametrize(
    ("background", "message"),
    [
        (np.array([1.0, 2.0, 3.0]), "two-dimensional"),
        (np.empty((0, 3)), "at least one background sample"),
        (np.zeros((2, 2)), "same number of features"),
    ],
)
def test_get_shap_explainer_rejects_invalid_background_shape(background, message):
    with pytest.raises(ValueError, match=message):
        get_shap_explainer(DummyEstimator(), np.zeros((4, 3)), X_background=background)


def test_get_shap_explainer_rejects_non_numeric_background():
    background = pd.DataFrame({"a": ["cat"], "b": [1], "c": [2]})

    with pytest.raises(TypeError, match="must contain numeric data"):
        get_shap_explainer(
            DummyEstimator(),
            np.zeros((4, 3)),
            X_background=background,
        )
