"""Воспроизвести E3 и проверить чувствительность к разбиению и размеру train."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.run_hw2 import CATS, NUMS, file_sha, ids_sha, make_model


def folds_for(y, n_splits, seed):
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, y))


def nested_subsets(train, y, fractions, seed, fold):
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
    by_class = [rng.permutation(train[y[train] == label]) for label in (0, 1)]
    return {
        f: np.sort(np.concatenate([idx[:round(f * len(idx))] for idx in by_class]))
        for f in fractions if f < 1
    }


def reference_scores(path, df, y, splits, reference):
    predictions = pd.read_csv(path, float_precision='round_trip')
    expected_fold = np.zeros(len(df), dtype=int)
    for i, (_, val) in enumerate(splits, 1):
        expected_fold[val] = i
    for column, values in [('id', df.id), ('target', y), ('fold', expected_fold)]:
        if not np.array_equal(predictions[column], values):
            raise ValueError(f'Столбец {column} в OOF не совпал с исходными данными и фолдами.')
    if not predictions.score.between(0, 1).all():
        raise ValueError('OOF содержит пропуски или значения вне [0, 1].')
    return predictions.score.to_numpy(), file_sha(path) == reference


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', type=Path, default=Path('data/train.csv'))
    cli.add_argument('--reference-oof', type=Path, help='Необязательный OOF E3 из ДЗ 2. Без него E3 обучается заново.')
    cli.add_argument('--output', type=Path, default=Path('outputs/hw3'))
    args = cli.parse_args()
    cfg = yaml.safe_load(Path('configs/hw3.yaml').read_text())
    hw2 = json.loads(Path('reports/hw2/manifest.json').read_text())
    model_cfg = yaml.safe_load(Path('configs/hw2.yaml').read_text())
    base_cfg = yaml.safe_load(Path('configs/baseline.yaml').read_text())
    input_sha = file_sha(args.data)
    if model_cfg != hw2['config'] or input_sha != hw2['input_sha256']:
        raise ValueError('Конфигурация или хеш train.csv отличаются от ДЗ 2.')
    df = pd.read_csv(args.data)
    y = df.satisfaction.astype(str).str.lower().map({'true': 1, 'false': 0}).to_numpy()
    if set(np.unique(y)) != {0, 1} or not df.id.is_unique:
        raise ValueError('Ожидаются уникальные ID и два класса True/False.')
    X = df[NUMS + CATS]
    seed42, *new_seeds = cfg['split_seeds']
    if seed42 != model_cfg['split_seed']:
        raise ValueError('Исходное разбиение должно совпасть с ДЗ 2.')
    splits = folds_for(y, model_cfg['n_splits'], seed42)
    for (tr, va), item in zip(splits, hw2['partitions'], strict=True):
        if (ids_sha(df.id.iloc[tr]), ids_sha(df.id.iloc[va])) != (item['train_ids_sha256'], item['val_ids_sha256']):
            raise ValueError('Состав исходного фолда отличается от ДЗ 2.')
    summary = pd.read_csv('reports/hw2/summary.csv').set_index('experiment')
    reference_sha = summary.loc['E3', 'oof_sha256']
    saved = pd.read_csv('reports/hw2/folds.csv').query("experiment == 'E3'").set_index('fold')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f'Каталог {out} не пуст. Укажите новый каталог через --output.')

    rows = []

    def record(seed, fraction, fold, tr, va, scores):
        if not np.isfinite(scores).all():
            raise ValueError('Модель вернула некорректные значения.')
        row = dict(split_seed=seed, train_fraction=fraction, fold=fold, n_train=len(tr),
                   n_val=len(va), n_train_positive=int(y[tr].sum()),
                   roc_auc=roc_auc_score(y[va], scores), ap=average_precision_score(y[va], scores),
                   train_ids_sha256=ids_sha(df.id.iloc[tr]), val_ids_sha256=ids_sha(df.id.iloc[va]))
        rows.append(row)
        print(f"seed={seed}, train={fraction:.0%}, фолд {fold}, ROC-AUC={row['roc_auc']:.8f}", flush=True)

    def fit(tr, va):
        model = make_model(base_cfg, model_cfg, cfg['features'])
        model.fit(X.iloc[tr], y[tr])
        return model.predict_proba(X.iloc[va])[:, 1]

    if args.reference_oof:
        scores, identical = reference_scores(args.reference_oof, df, y, splits, reference_sha)
        if not identical:
            raise ValueError('Хеш OOF не совпал с ДЗ 2. Запустите без --reference-oof.')
        source = 'hw2_oof'
    else:
        scores = np.full(len(df), np.nan)
        for tr, va in splits:
            scores[va] = fit(tr, va)
        source = 'retrained'
    fold_id = np.zeros(len(df), dtype=int)
    for i, (tr, va) in enumerate(splits, 1):
        fold_id[va] = i
        record(seed42, 1.0, i, tr, va, scores[va])
        for metric, column in [('roc_auc', 'roc_auc'), ('ap', 'average_precision')]:
            if abs(rows[-1][metric] - saved.loc[i, column]) > cfg['reference_auc_tolerance']:
                raise ValueError(f'Метрика {metric} исходного фолда {i} отличается от ДЗ 2.')
    pd.DataFrame(dict(id=df.id, target=y, fold=fold_id, score=scores)).to_csv(out / 'reference_oof.csv', index=False)
    for seed in new_seeds:
        for i, (tr, va) in enumerate(folds_for(y, model_cfg['n_splits'], seed), 1):
            record(seed, 1.0, i, tr, va, fit(tr, va))
    for i, (tr, va) in enumerate(splits, 1):
        subsets = nested_subsets(tr, y, cfg['train_fractions'], cfg['subset_seed'], i)
        for fraction, subset in subsets.items():
            record(seed42, fraction, i, subset, va, fit(subset, va))
    pd.DataFrame(rows).to_csv(out / 'robustness_folds.csv', index=False)
    metadata = dict(config=cfg, input_sha256=input_sha, hw2_e3_oof_sha256=reference_sha,
                    reference_source=source, report_oof_sha256=file_sha(out / 'reference_oof.csv'))
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Результаты сохранены в {out}.')


if __name__ == '__main__':
    main()
