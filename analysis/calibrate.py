"""Evaluate calibration on fixed test soils using seven sets of saved predictions."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'software'))
from diagnostics import choose, fit_calibration, predict, soil_errors


def main():
    data = pd.read_csv(ROOT / 'data/soil_records.csv')
    identity = pd.read_csv(ROOT / 'data/soil_identity.csv')
    data = data.merge(identity[['row_id', 'chen_soil_id']], on='row_id', validate='one_to_one')
    protocol = json.loads((ROOT / 'config/calibration_splits.json').read_text())
    models = protocol['models']
    saved = pd.read_csv(ROOT / 'data/predictions/calibration_predictions.csv', float_precision='round_trip')
    reference = pd.read_csv(ROOT / 'data/predictions/calibration_reference.csv', float_precision='round_trip')
    if saved.duplicated(['target_index', 'model', 'row_id']).any():
        raise ValueError('Duplicate saved prediction keys')
    rows = []
    checked = 0
    for task in sorted(protocol['tasks'], key=lambda t: t['target_index']):
        frame = data[data.study_id.eq(task['target_study'])].copy()
        for model in models:
            predictions = saved[saved.target_index.eq(task['target_index']) & saved.model.eq(model)]
            predictions = predictions[['row_id', 'source_pred']].rename(columns={'source_pred': 'pred_' + model})
            frame = frame.merge(predictions, on='row_id', validate='one_to_one', how='left')
            if frame['pred_' + model].isna().any():
                raise ValueError('Missing source predictions')
        frame['soil'] = frame.chen_soil_id.astype(str)
        frame['y'] = frame.target_log_kd
        for scenario in task['scenarios']:
            known = frame[frame.row_id.isin(scenario['selected_rows'])].copy()
            audit = frame[frame.row_id.isin(scenario['audit_rows'])].copy()
            if set(known.soil) & set(audit.soil):
                raise ValueError('Calibration and test soils overlap')
            if len(known) != len(scenario['selected_rows']) or len(audit) != len(scenario['audit_rows']):
                raise ValueError('Partition rows do not belong to the target study')
            for model in models + ['select_across_seven']:
                ref = 'pred_extratrees' if model == 'select_across_seven' else 'pred_' + model
                candidates = ['pred_' + m for m in models] if model == 'select_across_seven' else [ref]
                chosen = choose(known, candidates, ref)['selected']
                variants = {
                    'source': audit[ref].to_numpy(),
                    'always_offset': predict(audit, ref, fit_calibration(known, ref, 'offset_median')),
                    'always_affine': predict(audit, ref, fit_calibration(known, ref, 'affine_shrink')),
                    'tool_rule': predict(audit, chosen['model'], chosen['params']),
                }
                for strategy, prediction in variants.items():
                    rows.append({'study': task['target_study'], 'target_index': task['target_index'],
                                 'scenario': scenario['id'], 'budget': scenario['budget'],
                                 'model': model, 'strategy': strategy,
                                 'mae': float(soil_errors(audit, prediction).mean()),
                                 'selected_model': chosen['model'], 'selected_method': chosen['method'],
                                 'audit_soils': audit.soil.nunique()})
                if model == 'gp_matern':
                    expected = reference[reference.scenario.eq(scenario['id'])].set_index('row_id').loc[audit.row_id]
                    np.testing.assert_allclose(variants['always_affine'], expected.y_pred.to_numpy(), rtol=0, atol=2e-12)
                    checked += 1
        print(f"Completed target {task['target_index']}: {task['target_study']}", flush=True)
    results = pd.DataFrame(rows)
    per_study = results.groupby(['budget', 'model', 'study', 'strategy'], as_index=False).mae.mean()
    summaries = []
    for (budget, model), frame in per_study.groupby(['budget', 'model']):
        pivot = frame.pivot(index='study', columns='strategy', values='mae')
        summaries.append({'budget': int(budget), 'model': model, 'studies': len(pivot),
                          'source_mae': float(pivot.source.mean()), 'affine_mae': float(pivot.always_affine.mean()),
                          'tool_mae': float(pivot.tool_rule.mean()),
                          'tool_minus_affine': float((pivot.tool_rule - pivot.always_affine).mean()),
                          'tool_worse_than_affine_studies': int((pivot.tool_rule > pivot.always_affine + 1e-12).sum()),
                          'tool_worse_than_source_studies': int((pivot.tool_rule > pivot.source + 1e-12).sum())})
    out = ROOT / 'results/calibration'
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / 'partitions.csv', index=False)
    per_study.to_csv(out / 'per_study.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    print(f'Checked {checked} Gaussian process calibration partitions against saved results (tolerance 2e-12).')
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
