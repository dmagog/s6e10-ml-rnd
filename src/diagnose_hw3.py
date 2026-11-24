"""Подготовить агрегаты и графики по завершённому запуску ДЗ 3."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.run_hw2 import file_sha


TRAVEL = {'Business travel': 'Деловая', 'Personal Travel': 'Личная'}
GROUPS = [('Type of Travel', TRAVEL), ('Class', {'Business': 'Business', 'Eco': 'Eco', 'Eco Plus': 'Eco Plus'})]
PAIR = ['Seat comfort', 'Inflight entertainment']
BOARDING = 'Online boarding'
BANDS = ['0–2', '3', '4–5']


def diagnose(data, pred, cfg):
    df = data.join(pred[['fold', 'target', 'score']])
    threshold = cfg['threshold']
    df['prediction'] = (df.score >= threshold).astype(int)
    df['error'] = df.prediction != df.target
    df['kind'] = np.where(df.target == 1, 'FN', 'FP')
    near_lo, near_hi = cfg['near_bounds']
    far_lo, far_hi = cfg['extreme_bounds']
    df['band'] = np.select(
        [(df.score >= near_lo) & (df.score < near_hi), (df.score <= far_lo) | (df.score >= far_hi)],
        ['Около порога', 'Край шкалы'], default='Между диапазонами')
    errors = df[df.error]
    counts = errors.groupby(['kind', 'band']).size().rename('n').reset_index()
    counts['share_of_errors'] = counts.n / len(errors)
    for col in [*PAIR, BOARDING]:
        df[col + '_band'] = pd.cut(df[col], [-1, 2, 3, 5], labels=BANDS)
    errors = df[df.error]
    examples, local_examples = [], []
    for kind in ('FP', 'FN'):
        for band in ('Около порога', 'Край шкалы'):
            group = errors[(errors.kind == kind) & (errors.band == band)]
            case = group.assign(gap=(group.score - group.score.median()).abs()).sort_values(['gap', 'id']).iloc[0]
            similar = df[(df.target == case.target) & (df['Type of Travel'] == case['Type of Travel']) &
                         (df.Class == case.Class)]
            wrong = similar[similar.error & (similar.band == band)]
            correct = similar[~similar.error]
            def matching(frame):
                mask = (frame[PAIR[0] + '_band'] == case[PAIR[0] + '_band'])
                mask &= (frame[PAIR[1] + '_band'] == case[PAIR[1] + '_band'])
                return int(mask.sum())
            row = dict(example=len(examples) + 1, error=kind, band=band,
                       target=int(case.target), score=round(case.score, 3),
                       travel=TRAVEL[case['Type of Travel']], service_class=case.Class,
                       seat_band=str(case[PAIR[0] + '_band']), entertainment_band=str(case[PAIR[1] + '_band']),
                       same_pattern_errors=matching(wrong), n_errors=len(wrong),
                       same_pattern_correct=matching(correct), n_correct=len(correct))
            if kind == 'FN' and band == 'Край шкалы':
                key = BOARDING + '_band'
                row.update(boarding_band=str(case[key]),
                           boarding_errors=int((wrong[key] == case[key]).sum()),
                           boarding_correct=int((correct[key] == case[key]).sum()))
            examples.append(row)
            local_examples.append({**row, 'id': int(case.id), 'fold': int(case.fold)})
    slices, slice_folds = [], []
    for column, names in GROUPS:
        for value, label in names.items():
            group = df[df[column] == value]
            fold_aucs = []
            for i, val in group.groupby('fold'):
                auc = roc_auc_score(val.target, val.score)
                fold_aucs.append(auc)
                slice_folds.append(dict(slice=column, group=label, fold=i, n=len(val),
                                        positive=int(val.target.sum()), roc_auc=auc))
            pos = int(group.target.sum())
            neg = len(group) - pos
            fn = int((group.error & (group.target == 1)).sum())
            fp = int((group.error & (group.target == 0)).sum())
            slices.append(dict(slice=column, group=label, n=len(group), positive=pos, negative=neg,
                               positive_rate=pos / len(group), mean_fold_auc=float(np.mean(fold_aucs)),
                               min_fold_auc=min(fold_aucs), max_fold_auc=max(fold_aucs),
                               pooled_oof_auc=roc_auc_score(group.target, group.score),
                               fn=fn, fp=fp, fnr=fn / pos, fpr=fp / neg))
    examples = pd.DataFrame(examples).astype({'boarding_errors': 'Int64', 'boarding_correct': 'Int64'})
    return counts, examples, pd.DataFrame(local_examples), pd.DataFrame(slices), pd.DataFrame(slice_folds)


def figures(out, slices, fold, reference_seed):
    plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout='constrained')
    for ax, (column, _), title in zip(axes, GROUPS, ['Тип поездки', 'Класс обслуживания']):
        part = slices[slices.slice == column]
        ax.errorbar(
            part.mean_fold_auc, part.group,
            xerr=[part.mean_fold_auc - part.min_fold_auc, part.max_fold_auc - part.mean_fold_auc],
            fmt='o', color='#2456a6', ecolor='#778aab', elinewidth=1.5, capsize=4, zorder=3,
        )
        ax.set(title=title, xlabel='Средний ROC-AUC по фолдам', xlim=(min(0.7, slices.min_fold_auc.min() - 0.02), 1.0))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'.replace('.', ',')))
        ax.grid(axis='x', color='#e4e9ed', zorder=0)
        for label, v in zip(part.group, part.mean_fold_auc):
            ax.annotate(f'{v:.4f}'.replace('.', ','), (v, label), xytext=(-5, 10),
                        textcoords='offset points', ha='right', fontsize=9)
        ax.margins(y=.22)
    fig.text(0.5, 0.02, 'Точка: среднее. Отрезок: минимум и максимум по пяти фолдам.',
             ha='center', fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out / 'slices.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    seed_results = fold[fold.train_fraction == 1.0]
    for j, (seed, group) in enumerate(seed_results.groupby('split_seed', sort=False)):
        axes[0].scatter(np.full(len(group), j), group.roc_auc, color='#8c99a6', alpha=.8)
        axes[0].scatter(j, group.roc_auc.mean(), color='#2456a6', marker='D', s=55)
    axes[0].set_xticks(range(seed_results.split_seed.nunique()), seed_results.split_seed.unique())
    axes[0].set(title='Смена разбиения', xlabel='Seed разбиения', ylabel='ROC-AUC')
    axes[0].legend(handles=[
        Line2D([], [], color='#8c99a6', marker='o', linestyle='', label='Фолд'),
        Line2D([], [], color='#2456a6', marker='D', linestyle='', label='Среднее'),
    ], loc='lower right', frameon=False, fontsize=9)
    sizes = fold[fold.split_seed == reference_seed].pivot(index='fold', columns='train_fraction', values='roc_auc')
    delta = sizes.sub(sizes[1.0], axis=0)
    for i, (_, row) in enumerate(delta.iterrows()):
        axes[1].plot(delta.columns * 100, row, color='#8c99a6', alpha=.55, marker='o',
                     label='Фолд' if i == 0 else None)
    axes[1].plot(delta.columns * 100, delta.mean(), color='#2456a6', marker='D', linewidth=2.2,
                 label='Среднее')
    axes[1].set(title='Размер обучающей выборки', xlabel='Доля обучающей части, %',
                ylabel='Разница ROC-AUC относительно 100%')
    axes[1].legend(loc='lower right', frameon=False, fontsize=9)
    axes[1].set_xticks([20, 50, 100])
    for ax in axes:
        ax.grid(axis='y', color='#e4e9ed')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.4f}'.replace('.', ',').replace('-', '−')))
    fig.savefig(out / 'robustness.png', dpi=150)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', type=Path, default=Path('data/train.csv'))
    cli.add_argument('--run', type=Path, default=Path('outputs/hw3'))
    cli.add_argument('--report', type=Path)
    cli.add_argument('--check', action='store_true', help='Сверить рассчитанные таблицы с опубликованными.')
    args = cli.parse_args()
    out = args.report or args.run / 'report'
    published = Path('reports/hw3')
    if args.check and out.resolve() == published.resolve():
        cli.error('Для сверки сохраните новый отчёт в отдельный каталог.')
    manifest = json.loads((args.run / 'manifest.json').read_text())
    pred_path = args.run / 'reference_oof.csv'
    if file_sha(args.data) != manifest['input_sha256'] or file_sha(pred_path) != manifest['report_oof_sha256']:
        raise ValueError('Хеш данных или OOF отличается от манифеста.')
    cfg = manifest['config']
    data = pd.read_csv(args.data, usecols=['id', 'satisfaction', 'Type of Travel', 'Class', *PAIR, BOARDING])
    pred = pd.read_csv(pred_path, float_precision='round_trip')
    if not (data.id.equals(pred.id) and np.array_equal(data.satisfaction.astype(int), pred.target)):
        raise ValueError('ID или метки OOF не совпадают с train.csv.')

    fold = pd.read_csv(args.run / 'robustness_folds.csv')
    seed42 = cfg['split_seeds'][0]
    ref = fold[(fold.split_seed == seed42) & (fold.train_fraction == 1)].set_index('fold')
    current = fold[fold.split_seed == seed42].set_index('fold')
    if not current.val_ids_sha256.eq(ref.val_ids_sha256).all():
        raise ValueError('В проверке размера train изменились валидационные строки.')
    fold['delta_vs_full'] = np.where(fold.split_seed == seed42, fold.roc_auc - fold.fold.map(ref.roc_auc), np.nan)
    out.mkdir(parents=True, exist_ok=True)
    counts, examples, local_examples, slices, slice_folds = diagnose(data, pred, cfg)
    local_examples.to_csv(args.run / 'error_examples_local.csv', index=False)
    tables = [('error_counts', counts), ('error_examples', examples), ('slices', slices),
              ('slice_folds', slice_folds), ('robustness_folds', fold)]
    for name, table in tables:
        table.to_csv(out / f'{name}.csv', index=False)
    figures(out, slices, fold, seed42)
    print(f'Таблицы и графики сохранены в {out}.')
    if args.check:
        for name, _ in tables:
            filename = f'{name}.csv'
            try:
                pd.testing.assert_frame_equal(
                    pd.read_csv(out / filename), pd.read_csv(published / filename),
                    check_exact=False, rtol=0, atol=cfg['reference_auc_tolerance'],
                )
            except AssertionError as exc:
                raise SystemExit(f'Таблица {filename} отличается от опубликованной.\n{exc}') from exc
        print('Все пять таблиц совпали с опубликованными в пределах допуска.')


if __name__ == '__main__':
    main()
