"""Собрать небольшие проверяемые таблицы и график после полного прогона."""
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

source = Path('outputs/hw2')
out = Path('reports/hw2')
out.mkdir(parents=True, exist_ok=True)
for name in ['summary.csv', 'folds.csv', 'comparisons.csv', 'manifest.json']:
    shutil.copyfile(source / name, out / name)
summary = pd.read_csv(source / 'summary.csv').set_index('experiment')
comp = pd.read_csv(source / 'comparisons.csv').set_index('comparison')
folds = pd.read_csv(source / 'folds.csv')
raw = pd.read_csv('data/train.csv', usecols=['id', 'Type of Travel'])
subsets = []
for experiment in ['E0', 'E1', 'E4']:
    pred = pd.read_csv(source / f'{experiment}_oof.csv')
    if not np.array_equal(pred.id, raw.id):
        raise ValueError('Порядок строк предсказаний не совпадает с train')
    for name in ['Business travel', 'Personal Travel']:
        mask = raw['Type of Travel'] == name
        subsets.append({'experiment': experiment, 'type_of_travel': name, 'n': int(mask.sum()),
                        'positive_rate': float(pred.loc[mask, 'target'].mean()),
                        'pooled_oof_auc': float(roc_auc_score(pred.loc[mask, 'target'], pred.loc[mask, 'score']))})
pd.DataFrame(subsets).to_csv(out / 'slices.csv', index=False)
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
colors = ['#8291a9', '#2456a6', '#4182bc', '#5ba1b1', '#d97842']
bars = ax[0].bar(summary.index, summary['mean_fold_auc'], color=colors)
ax[0].bar_label(bars, labels=[f'{v:.5f}' for v in summary['mean_fold_auc']], padding=3)
ax[0].set_ylim(.89, .98)
ax[0].set(title='Средний ROC-AUC по фолдам', ylabel='ROC-AUC', xlabel='Эксперимент')
names = ['both_features', 'ablate_A', 'ablate_B']
labels = ['A+B против исходных', 'Вклад A при наличии B', 'Вклад B при наличии A']
for j, (name, label) in enumerate(zip(names, labels)):
    row = comp.loc[name]
    vals = np.array([row[f'fold_{i}_delta'] for i in range(1, 6)]) * 10000
    ax[1].scatter(vals, np.full(5, j), color='#8c99a6', s=28, alpha=.8)
    ax[1].scatter(vals.mean(), j, color='#bf572a', s=58, marker='D', zorder=4)
ax[1].axvline(0, color='#777777', linestyle='--', linewidth=1)
ax[1].set_yticks(range(len(labels)), labels=labels)
ax[1].invert_yaxis()
ax[1].set(title='Парная разница ROC-AUC', xlabel='Разница × 10 000; точки: фолды, ромб: среднее')
fig.tight_layout()
fig.savefig(out / 'comparison.png', dpi=150)
plt.close(fig)
print('Отчётные таблицы и график обновлены.')
