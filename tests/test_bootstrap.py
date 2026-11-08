"""Проверка bootstrap ROC-AUC на малой выборке со связками."""
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from src.bootstrap import bootstrap_auc


class BootstrapAucTest(unittest.TestCase):
    def test_matches_direct_stratified_resampling_with_ties(self):
        target = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        score = np.array([0.3, 0.3, 0.7, 0.5, 0.1, 0.7, 0.5, 0.9])
        seed, repetitions = 37, 100
        result = bootstrap_auc(target, score, repetitions=repetitions, seed=seed)
        order = np.argsort(score, kind='stable')
        sorted_target, sorted_score = target[order], score[order]
        negatives = np.flatnonzero(sorted_target == 0)
        positives = np.flatnonzero(sorted_target == 1)
        rng = np.random.default_rng(seed)
        direct = []
        for _ in range(repetitions):
            idx = np.r_[negatives[rng.integers(len(negatives), size=len(negatives))],
                        positives[rng.integers(len(positives), size=len(positives))]]
            direct.append(roc_auc_score(sorted_target[idx], sorted_score[idx]))
        np.testing.assert_allclose([result['lower'], result['upper']],
                                   np.quantile(direct, [.025, .975]), rtol=0, atol=1e-14)
        self.assertAlmostEqual(result['point_auc'], roc_auc_score(target, score))

    def test_rejects_single_class_and_nonfinite_scores(self):
        for target, score in [([1, 1], [0.1, 0.9]), ([0, 1], [0.1, np.nan])]:
            with self.assertRaises(ValueError):
                bootstrap_auc(target, score, repetitions=2)


if __name__ == '__main__':
    unittest.main()
