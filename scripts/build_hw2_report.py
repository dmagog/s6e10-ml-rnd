"""Собрать небольшие проверяемые таблицы и график после полного прогона."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='outputs/hw2', help='Каталог завершённого прогона')
parser.add_argument('--report-dir', help='Каталог отчёта; по умолчанию report внутри каталога прогона')
args = parser.parse_args()
source = Path(args.output)
out = Path(args.report_dir) if args.report_dir else source / 'report'
subprocess.run([sys.executable, '-m', 'src.verify_hw2', '--output', str(source)], check=True)
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
slices = pd.DataFrame(subsets)
slices.to_csv(out / 'slices.csv', index=False)
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
colors = ['#8291a9', '#2456a6', '#4182bc', '#5ba1b1', '#d97842']
ax[0].scatter(summary.index, summary['mean_fold_auc'], color=colors, s=65)
for x, value in zip(summary.index, summary['mean_fold_auc']):
    ax[0].annotate(f'{value:.5f}'.replace('.', ','), (x, value), xytext=(0, 8),
                   textcoords='offset points', ha='center')
low, high = summary['mean_fold_auc'].min(), summary['mean_fold_auc'].max()
margin = max((high - low) * 0.16, 0.002)
ax[0].set_ylim(max(0, low - margin), min(1, high + margin))
ax[0].set_xlim(-0.4, len(summary) - 0.6)
ax[0].set(title='Средний ROC-AUC по фолдам', ylabel='ROC-AUC', xlabel='Эксперимент')
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.2f}'.replace('.', ',')))
names = ['both_features', 'ablate_A', 'ablate_B']
labels = ['A + B против исходных', 'Добавление A при наличии B', 'Добавление B при наличии A']
for j, (name, label) in enumerate(zip(names, labels)):
    row = comp.loc[name]
    vals = np.array([row[f'fold_{i}_delta'] for i in range(1, 6)]) * 10000
    ax[1].scatter(vals, np.full(5, j), color='#8c99a6', s=28, alpha=.8)
    ax[1].scatter(vals.mean(), j, color='#bf572a', s=58, marker='D', zorder=4)
ax[1].axvline(0, color='#777777', linestyle='--', linewidth=1)
ax[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}'.replace('.', ',')))
ax[1].set_yticks(range(len(labels)), labels=labels)
ax[1].invert_yaxis()
ax[1].set(title='Парная разница ROC-AUC', xlabel='Разница × 10 000; точки: фолды, ромб: среднее')
fig.tight_layout()
fig.savefig(out / 'comparison.png', dpi=150)
plt.close(fig)

# Срезы рассчитаны по объединённым OOF-прогнозам; точки не являются интервалами.
fig, ax = plt.subplots(figsize=(9, 4))
groups = ['Business travel', 'Personal Travel']
models = [('E0', 'Логистическая регрессия (E0)', '#8291a9', -0.18),
          ('E1', 'LightGBM (E1)', '#2456a6', 0),
          ('E4', 'LightGBM + A + B (E4)', '#d97842', 0.18)]
for experiment, label, color, offset in models:
    values = [slices.query('experiment == @experiment and type_of_travel == @group').iloc[0]
              for group in groups]
    ys = np.arange(len(groups)) + offset
    xs = [row['pooled_oof_auc'] for row in values]
    ax.scatter(xs, ys, color=color, s=48, label=label, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f'{x:.4f}'.replace('.', ','), (x, y), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=9)
labels = []
for group, name in zip(groups, ['Деловые поездки', 'Личные поездки']):
    row = slices[slices.type_of_travel == group].iloc[0]
    count = f"{int(row['n']):,}".replace(",", " ")
    positive_rate = f"{row['positive_rate']:.1%}".replace(".", ",")
    labels.append(f"{name}\nn = {count}; положительных = {positive_rate}")
ax.set_yticks(range(len(groups)), labels)
ax.invert_yaxis()
ax.set_xlim(0.5, 1.0)
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}'.replace('.', ',')))
ax.set_xlabel('ROC-AUC по объединённым OOF-прогнозам')
ax.set_title('Качество ранжирования по типу поездки')
ax.grid(axis='x', color='#e1e5ea', linewidth=0.8)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(out / 'travel_slices.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Отчётные таблицы и графики обновлены.')
