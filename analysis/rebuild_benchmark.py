"""Recompute the published-data benchmark from frozen row-level predictions."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    frame = pd.read_csv(ROOT / 'data/predictions/benchmark_predictions.csv', float_precision='round_trip')
    if frame.duplicated(['model', 'mode', 'seed', 'row_id']).any():
        raise ValueError('Duplicate prediction keys')
    frame['absolute_error'] = abs(frame.y_pred - frame.y_true)
    per_study = (frame.groupby(['model', 'mode', 'study_id', 'seed']).absolute_error.mean()
                 .groupby(['model', 'mode', 'study_id']).mean().reset_index(name='mae'))
    studies = sorted(frame.study_id.unique())
    if len(studies) != 47:
        raise ValueError('Unexpected benchmark study count')
    indices = np.random.default_rng(20260924).integers(0, len(studies), (10000, len(studies)))
    rows = []
    for (model, mode), group in per_study.groupby(['model', 'mode']):
        vector = group.set_index('study_id').loc[studies, 'mae'].to_numpy()
        low, high = np.quantile(vector[indices].mean(axis=1), [.025, .975])
        rows.append({'model': model, 'mode': mode, 'mae': vector.mean(), 'ci_low': low, 'ci_high': high})
    summary = pd.DataFrame(rows).set_index(['model', 'mode']).sort_index()
    original = pd.read_csv(ROOT / 'data/predictions/benchmark_reference.csv',
                           float_precision='round_trip').set_index(['model', 'mode']).sort_index()
    # The original figure table contains tuned, default, and historical settings.
    for column in ['mae', 'ci_low', 'ci_high']:
        np.testing.assert_allclose(summary.loc[original.index, column], original[column], rtol=0, atol=2e-12)
    out = ROOT / 'results/benchmark'
    out.mkdir(parents=True, exist_ok=True)
    per_study.to_csv(out / 'per_study.csv', index=False)
    summary.reset_index().to_csv(out / 'summary.csv', index=False)
    print(f'Reproduced {len(summary)} model/setting rows (18 models and the median reference) on 47 studies.')
    print('MAE and bootstrap interval endpoints match the original figure table within 2e-12.')


if __name__ == '__main__':
    main()
