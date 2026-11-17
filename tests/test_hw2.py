import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.run_hw2 import AddInteractions


class InteractionTest(unittest.TestCase):
    def test_each_component_is_row_local_and_ablatable(self):
        cfg = yaml.safe_load(Path('configs/hw2.yaml').read_text())['features']
        X = pd.DataFrame({'Inflight wifi service': [2.0, np.nan], 'Online boarding': [3.0, 4.0],
                          'Seat comfort': [4.0, 1.0], 'Inflight entertainment': [5.0, 2.0]})
        before = X.copy(deep=True)
        a = AddInteractions([cfg['A']]).fit(X).transform(X)
        b = AddInteractions([cfg['B']]).fit(X).transform(X)
        both = AddInteractions([cfg['A'], cfg['B']]).fit(X).transform(X)
        pd.testing.assert_frame_equal(X, before)
        self.assertEqual(both.loc[0, cfg['A']['name']], 6.0)
        self.assertTrue(pd.isna(both.loc[1, cfg['A']['name']]))
        self.assertEqual(both.loc[0, cfg['B']['name']], 20.0)
        pd.testing.assert_frame_equal(both.drop(columns=cfg['B']['name']), a)
        pd.testing.assert_frame_equal(both.drop(columns=cfg['A']['name']), b)


if __name__ == '__main__':
    unittest.main()
