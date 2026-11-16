# Improving Cross-Study Predictions of PFAS Sorption in Soils

Data, model training, prediction analysis, and a local graphical tool for PFAS soil–water distribution coefficients.

## Contents

| Directory | Contents |
|---|---|
| `data/` | Two measured-data tables, soil identifiers, response provenance, and publication DOIs |
| `data/predictions/` | Saved benchmark and calibration predictions, with numerical reference results |
| `config/` | Input definitions, model parameters, and fixed training/validation/test partitions |
| `models/` | Model implementations and required third-party sources/licenses |
| `training/` | One training entry point |
| `analysis/` | Benchmark scoring and calibration analysis |
| `software/` | Local graphical application, example input, and scientific tests |

Dataset sizes, units, columns, and source DOIs are described in [data/README.md](data/README.md). Generated outputs go to `results/`, which is excluded from Git.

## Install

Python 3.12 is recommended. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The base installation supports scoring, calibration, the graphical tool, and scikit-learn models. Install the additional packages in `requirements-models.txt` to train boosting, neural-network, or pretrained models. TabPFN also requires separately obtained, version-specific weights under the provider's license.

## Train and compare inputs

```sh
# Ridge regression on all five recorded publication-grouped folds.
python training/train.py --model ridge

# Select parameters using inner training-study folds; evaluate fold 0.
python training/train.py --model extratrees --mode both --fold 0

# Repeat the same split without organic carbon as an input.
python training/train.py --model ridge --drop corg_percent
```

Each run writes predictions, study-level errors, and fitted settings. `--mode fixed` uses candidate 0 from `config/models.json`; `--mode tuned` selects among 12 candidates using three inner study-grouped folds; `--mode both` produces both. Neural models determine training duration on these inner folds even in fixed mode. The test-study responses are used only for evaluation. Preprocessing is fitted on training rows.

Omit `--seed` to use the recorded seeds. Use `--fold 0` through `--fold 4` to run folds separately. Choose a new `--output` directory to repeat a completed run. Input-removal runs refit the model after dropping the named columns; compare their predictions on identical test rows.

| Model category | `--model` values |
|---|---|
| Linear and additive | `ridge`, `elastic_net`, `additive_spline` |
| Nearest neighbors | `knn` |
| Trees and tree ensembles | `decision_tree`, `random_forest`, `extratrees`, `xgboost`, `lightgbm`, `catboost` |
| Kernel methods | `svr_rbf`, `gp_matern` |
| Neural networks trained on the task data | `mlp`, `resnet`, `ft_transformer`, `tabm` |
| Pretrained tabular models | `tabpfn_v2`, `tabpfn_v3` |

TabPFN uses native preprocessing and eight ensemble members, with no hyperparameter search. Pass `--mode fixed --checkpoint /path/to/version-specific.ckpt`; the checkpoint SHA-256 must match `config/models.json`. `--device cuda` is available for TabPFN. The other adapters run on CPU. Model weights are not included. The median reference in the saved benchmark is a separate non-learning baseline.

## Recompute results from saved predictions

```sh
python analysis/rebuild_benchmark.py
python analysis/calibrate.py
```

The benchmark reconstructs 35 model/setting rows for 18 models and the median reference across 47 studies. It averages errors within studies and seeds before averaging across studies, and checks the study-bootstrap intervals against saved reference values.

Calibration analysis uses seven models, 19 studies, and 180 fixed partitions. It fits corrections on measured calibration soils and evaluates on separate test soils. Outputs distinguish original predictions, median-offset correction, regularized calibration, and exploratory validation-based selection. The calibration replay checks implementation consistency; it is not independent validation of the selection policy. It does not refit the source models.

These commands cover the compact benchmark, input-removal training, and calibration workflows. Saved predictions reproduce the recorded numerical comparisons; fresh fits can differ with package versions. The repository does not regenerate every supplementary analysis or manuscript figure.

## Graphical tool

```sh
python software/app.py
```

Open <http://127.0.0.1:8766>, load the example, or upload a CSV, TSV, or XLSX file. The tool diagnoses bias, calibrates existing predictions, and evaluates accuracy and soil ranking when separate measured test soils are available. It requires numerical model predictions; no model file is uploaded. It does not generate predictions directly from soil descriptors or certify measurement quality. See [software/README.md](software/README.md).

```sh
PYTHONPATH=software python -m unittest discover -s software/tests -v
```

## Attribution

Upstream data and pretrained weights retain their original terms. Vendored TabM 0.0.3, RTDL numerical embeddings 0.0.12, and RTDL revisiting models 0.0.2 retain their license notices in `models/vendor/`. A project-wide reuse license has not been selected.
