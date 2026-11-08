"""Независимо пересчитать метрики из сохранённых OOF-предсказаний."""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

p = argparse.ArgumentParser()
p.add_argument('oof')
a = p.parse_args()
df = pd.read_csv(a.oof)
if df['id'].duplicated().any() or df[['fold', 'target', 'score']].isna().any().any():
    raise ValueError('Некорректный OOF-файл')
folds = [roc_auc_score(g.target, g.score) for _, g in df.groupby('fold')]
print(json.dumps({'n': len(df), 'fold_auc': folds, 'mean_auc': float(np.mean(folds)),
                  'std_auc': float(np.std(folds)), 'oof_auc': roc_auc_score(df.target, df.score),
                  'oof_average_precision': average_precision_score(df.target, df.score)}, indent=2))
