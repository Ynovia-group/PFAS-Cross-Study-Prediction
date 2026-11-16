"""Fit models with publication-grouped validation and training-only preprocessing."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.representative import DEEP, fit_predict, study_mae


def read_json(name):
    return json.loads((ROOT / 'config' / name).read_text())


def partition(frame, split, test_key):
    train_ids, test_ids = split['train_row_ids'], split[test_key]
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError('Duplicate row identifiers in a split')
    train = frame[frame.row_id.isin(train_ids)].copy()
    test = frame[frame.row_id.isin(test_ids)].copy()
    if len(train) != len(train_ids) or len(test) != len(test_ids):
        raise ValueError('Split contains rows outside its permitted parent dataset')
    for column in ['row_id', 'study_id', 'leakage_group_id']:
        if set(train[column]) & set(test[column]):
            raise ValueError(f'Training and evaluation share {column}')
    return train, test


def select(spec, train, fold, contract, mode):
    candidates = spec['candidates'] if mode != 'fixed' else spec['candidates'][:1]
    scores = []
    for index, params in enumerate(candidates):
        values, epochs = [], []
        for split in fold['inner_folds']:
            inner_train, validation = partition(train, split, 'validation_row_ids')
            predictions, details = fit_predict(
                spec, params, 11, inner_train, validation[contract['feature_columns']], contract,
                validation=(validation[contract['target']].to_numpy(), validation.study_id.to_numpy())
                if spec['model'] in DEEP else None,
            )
            values.append(study_mae(validation[contract['target']], predictions, validation.study_id))
            if spec['model'] in DEEP:
                epochs.append(details['best_epoch'])
        scores.append({'candidate': index, 'parameters': params, 'score': float(np.mean(values)),
                       'inner_scores': values,
                       'epochs': max(1, int(np.rint(np.median(epochs)))) if epochs else None})
    selected = min(scores, key=lambda item: (item['score'], item['candidate']))['candidate']
    return selected, scores


def run(args):
    contract = read_json('features.json')
    unknown = set(args.drop) - set(contract['feature_columns'])
    if unknown:
        raise ValueError(f'Unknown input columns: {sorted(unknown)}')
    for key in ['feature_columns', 'numeric_features', 'categorical_features']:
        contract[key] = [c for c in contract[key] if c not in args.drop]
    if not contract['numeric_features']:
        raise ValueError('Retain at least one numerical input')
    frame = pd.read_csv(ROOT / 'data/soil_records.csv', float_precision='round_trip').sort_values('row_id')
    if not frame.row_id.is_unique or not np.isfinite(frame[contract['target']]).all():
        raise ValueError('Training records need unique identifiers and finite response values')
    spec = copy.deepcopy(read_json('models.json')[args.model])
    is_pretrained = args.model.startswith('tabpfn_')
    if is_pretrained and (args.mode != 'fixed' or args.checkpoint is None):
        raise ValueError('TabPFN requires --mode fixed and an explicit --checkpoint file')
    view = 'all_inputs' if not args.drop else 'without_' + '_'.join(sorted(args.drop))
    output = args.output.resolve() / args.model / view / args.mode
    if output == ROOT or ROOT / 'data' in output.parents or ROOT / 'config' in output.parents:
        raise ValueError('Write generated results outside source data and configuration directories')
    folds = read_json('benchmark_splits.json')['folds']
    for fold in folds:
        number = fold['outer_fold']
        if args.fold is not None and number != args.fold:
            continue
        dest = output / f'fold_{number}'
        if dest.exists():
            raise FileExistsError(f'Use a new --output directory; results already exist at {dest}')
        train, test = partition(frame, fold, 'test_row_ids')
        if is_pretrained:
            selected, scores = 0, []
        elif args.mode != 'fixed' or args.model in DEEP:
            selected, scores = select(spec, train, fold, contract, args.mode)
        else:
            selected, scores = 0, []
        modes = ['fixed', 'tuned'] if args.mode == 'both' else [args.mode]
        parts, fits = [], []
        seeds = [args.seed] if args.seed is not None else spec['seeds']
        for index in sorted({0 if mode == 'fixed' else selected for mode in modes}):
            for seed in seeds:
                if is_pretrained:
                    from models.tabpfn import predict_native
                    pred, details = predict_native(spec, seed, train, test[contract['feature_columns']],
                                                   contract, args.checkpoint, args.device)
                else:
                    epochs = scores[index]['epochs'] if scores else None
                    pred, details = fit_predict(spec, spec['candidates'][index], seed, train,
                                                test[contract['feature_columns']], contract, epochs=epochs)
                fits.append({'candidate': index, 'seed': seed, **details})
                for mode in modes:
                    if index != (0 if mode == 'fixed' else selected):
                        continue
                    part = test[['row_id', 'study_id']].copy()
                    part['model'], part['mode'], part['seed'], part['fold'] = args.model, mode, seed, number
                    part['y_pred'], part['y_true'] = pred, test[contract['target']].to_numpy()
                    parts.append(part)
        predictions = pd.concat(parts, ignore_index=True)
        predictions['absolute_error'] = abs(predictions.y_pred - predictions.y_true)
        per_study = predictions.groupby(['mode', 'seed', 'study_id'], as_index=False).absolute_error.mean()
        per_study = per_study.rename(columns={'absolute_error': 'mae'})
        versions = {}
        for package in ['numpy', 'pandas', 'scikit-learn', 'torch', 'xgboost', 'lightgbm', 'catboost', 'tabpfn']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        dest.mkdir(parents=True)
        predictions.drop(columns='absolute_error').to_csv(dest / 'predictions.csv', index=False)
        per_study.to_csv(dest / 'per_study.csv', index=False)
        metadata = {'model': args.model, 'fold': number, 'inputs': contract['feature_columns'],
                    'selected_candidate': selected, 'inner_validation': scores, 'fits': fits,
                    'train_records': len(train), 'test_records': len(test), 'packages': versions,
                    'test_responses_used_for_selection': False}
        (dest / 'fit.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        print(f'{args.model}: fold {number}, {len(test)} test records -> {dest}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(read_json('models.json')), default='ridge')
    parser.add_argument('--mode', choices=['fixed', 'tuned', 'both'], default='fixed')
    parser.add_argument('--fold', type=int, choices=range(5), help='Omit to run all five folds')
    parser.add_argument('--seed', type=int, help='Omit to use the recorded model-specific seeds')
    parser.add_argument('--drop', nargs='+', default=[], help='Input columns to omit before fitting')
    parser.add_argument('--checkpoint', type=Path, help='Verified TabPFN V2 or V3 checkpoint')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='TabPFN execution device')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/training')
    run(parser.parse_args())
