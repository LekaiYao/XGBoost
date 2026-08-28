import numpy as np

from workflows.x_sideband_mass_efficiency import binomial_efficiency, threshold_map


def test_binomial_efficiency_and_empty_bin():
    efficiency, uncertainty = binomial_efficiency(np.array([5, 0, 0]), np.array([10, 4, 0]))
    assert np.allclose(efficiency[:2], [0.5, 0.0])
    assert np.allclose(uncertainty[:2], [np.sqrt(0.025), 0.0])
    assert np.isnan(efficiency[2])
    assert np.isnan(uncertainty[2])


def test_threshold_map_indexes_numeric_efficiencies():
    payload = {
        "thresholds": [
            {"target_efficiency": 0.2, "score_threshold": 0.9},
            {"target_efficiency": 0.3, "score_threshold": 0.8},
        ]
    }
    mapped = threshold_map(payload)
    assert mapped[0.2]["score_threshold"] == 0.9
    assert mapped[0.3]["score_threshold"] == 0.8
