"""Version-specific TabPFN inference with verified checkpoints and native inputs."""
import hashlib
import os
from pathlib import Path

import numpy as np


def predict_native(spec, seed, train, inputs, contract, checkpoint, device):
    path = Path(checkpoint).expanduser().resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec['checkpoint_sha256']:
        raise ValueError(f"Checkpoint does not match the recorded TabPFN {spec['version']} weights")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    if device == 'cpu' and len(train) > 1000:
        os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
    columns = contract['feature_columns']
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion[spec['version']], model_path=path, device=device,
        n_estimators=spec['n_estimators'], random_state=seed,
        categorical_features_indices=[columns.index(c) for c in contract['categorical_features']],
        fit_mode='low_memory', keep_cache_on_device=False, n_preprocessing_jobs=1,
        show_progress_bar=False,
    )
    model.fit(train[columns].copy(), train[contract['target']].to_numpy(np.float32))
    predictions = np.asarray(model.predict(inputs.copy()), dtype=float)
    if predictions.shape != (len(inputs),) or not np.isfinite(predictions).all():
        raise ValueError('Invalid TabPFN predictions')
    return predictions, {'version': spec['version'], 'checkpoint_sha256': digest,
                          'n_estimators': spec['n_estimators'], 'device': device}
