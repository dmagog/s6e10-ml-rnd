import unittest
import pandas as pd

from src.compare_hw2 import compare_folds


class CompareRunsTest(unittest.TestCase):
    def test_compares_all_folds_and_detects_changes(self):
        ref = pd.DataFrame([
            {'experiment': f'E{experiment}', 'fold': fold, 'roc_auc': 0.9, 'average_precision': 0.8}
            for experiment in range(5) for fold in range(1, 6)
        ])
        same = ref.sample(frac=1, random_state=42)
        self.assertTrue(compare_folds(ref, same, 1e-6)[1])
        changed = same.copy()
        changed.loc[changed.index[0], 'roc_auc'] += 0.00001
        self.assertFalse(compare_folds(ref, changed, 1e-6)[1])
        with self.assertRaises(ValueError):
            compare_folds(ref, same.iloc[:-1], 1e-6)
