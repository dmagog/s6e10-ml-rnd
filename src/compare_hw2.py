"""Сравнить метрики завершённого повтора с опубликованным снимком."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_folds(reference, current, atol):
    """Сравнить метрики каждого варианта на каждом фолде."""
    keys = ['experiment', 'fold']
    metrics = ['roc_auc', 'average_precision']
    for frame in (reference, current):
        if frame.duplicated(keys).any():
            raise ValueError('Повторяются пары эксперимент–фолд')
    paired = reference.merge(current, on=keys, how='outer', suffixes=('_reference', '_current'), indicator=True)
    if len(paired) != 25 or not (paired['_merge'] == 'both').all():
        raise ValueError('Наборы экспериментов и фолдов не совпадают')
    if set(paired.experiment) != {'E0', 'E1', 'E2', 'E3', 'E4'} or set(paired.fold) != set(range(1, 6)):
        raise ValueError('Не найдены все пять вариантов и пять фолдов')
    differences = {}
    for metric in metrics:
        values = paired[f'{metric}_current'] - paired[f'{metric}_reference']
        if not np.isfinite(values).all():
            raise ValueError(f'{metric}: некорректная метрика')
        differences[metric] = float(values.abs().max())
    return differences, all(value <= atol for value in differences.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/hw2')
    parser.add_argument('--reference', default='reports/hw2')
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0:
        parser.error('Допуск должен быть неотрицательным конечным числом')
    reference_dir = Path(args.reference)
    output_dir = Path(args.output)
    ref_manifest = json.loads((reference_dir / 'manifest.json').read_text())
    run_manifest = json.loads((output_dir / 'manifest.json').read_text())
    for key in ['input_sha256', 'config', 'partitions']:
        if ref_manifest[key] != run_manifest[key]:
            raise ValueError(f'Параметр {key} отличается от опубликованного')
    reference = pd.read_csv(reference_dir / 'folds.csv')
    current = pd.read_csv(output_dir / 'folds.csv')
    differences, matches = compare_folds(reference, current, args.atol)
    for metric, difference in differences.items():
        print(f'{metric}: максимальное абсолютное отличие по 25 фолдам = {difference:.10g}')
    print(f'Допуск для каждой метрики: {args.atol:g}')
    if not matches:
        raise SystemExit('Метрики отличаются от опубликованных выше допуска; проверьте окружение и результаты.')
    print('Все метрики фолдов совпадают с опубликованными в пределах допуска.')


if __name__ == '__main__':
    main()
