"""Drift statistics on distributions whose answer is known in advance."""

import numpy as np

from monitoring.drift import ALERT, OK, WARNING, categorical_psi, ks_distance, numeric_psi, psi_status, worst_status

rng = np.random.default_rng(0)
REFERENCE = rng.normal(0.6, 0.1, 2000)


def test_same_distribution_has_near_zero_drift() -> None:
    same = rng.normal(0.6, 0.1, 2000)
    assert numeric_psi(REFERENCE, same) < 0.05
    assert ks_distance(REFERENCE, same) < 0.05


def test_shifted_distribution_is_an_alert() -> None:
    darker = rng.normal(0.4, 0.1, 2000)  # e.g. images two standard deviations darker
    assert psi_status(numeric_psi(REFERENCE, darker)) == ALERT
    assert ks_distance(REFERENCE, darker) > 0.5


def test_constant_reference_still_detects_new_values() -> None:
    reference = [1.3333] * 300  # every TrashNet image is 4:3
    assert numeric_psi(reference, [1.3333] * 300) == 0.0
    assert psi_status(numeric_psi(reference, [0.75] * 300)) == ALERT  # portrait phone photos
    assert psi_status(numeric_psi(reference, [1.7778] * 300)) == ALERT  # 16:9 photos


def test_categorical_psi() -> None:
    reference = ["glass"] * 50 + ["paper"] * 50
    assert categorical_psi(reference, ["glass"] * 50 + ["paper"] * 50) == 0.0
    assert psi_status(categorical_psi(reference, ["glass"] * 90 + ["paper"] * 10)) == ALERT
    assert categorical_psi(reference, ["glass"] * 50 + ["trash"] * 50) > 0.25  # a class never seen before


def test_ks_distance_with_no_overlap_is_one() -> None:
    assert ks_distance([0.1, 0.2, 0.3], [0.7, 0.8, 0.9]) == 1.0


def test_status_thresholds() -> None:
    assert (psi_status(0.05), psi_status(0.1), psi_status(0.25), psi_status(0.3)) == (OK, WARNING, WARNING, ALERT)
    assert worst_status([OK, WARNING, OK]) == WARNING
    assert worst_status([OK, ALERT, WARNING]) == ALERT
    assert worst_status([OK, OK]) == OK
