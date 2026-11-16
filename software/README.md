# PFAS Prediction Diagnostics

A local graphical tool for evaluating and calibrating existing PFAS soil–water distribution-coefficient predictions. Start it from the repository root:

```sh
python software/app.py
```

Open <http://127.0.0.1:8766>. Files are processed in local memory. The application accepts CSV, TSV, and XLSX; it does not upload data to a remote service.

## Input

Use **Load the paper example** or upload a table containing:

| Column | Purpose |
|---|---|
| `study_id` | Study or independently calibrated group |
| `soil_id` | Soil identifier; all records for a soil must have the same role |
| `pfas` | Compound identifier |
| `condition_id` | Comparable experimental conditions for ranking |
| `role` | `update` for calibration, `audit` for separate testing, or `predict` for unmeasured soils |
| `observed` | Measured Kd or log10[Kd/(L/kg)], with the scale selected in the interface |
| `pred_*` | One or more model-prediction columns, with their scale selected separately |

Column names can be mapped in the interface. A blank upload template is available. Calibration and test rows need measured values; prediction-only rows must leave them blank. Predictions must have been generated without using measured Kd from the target study for training or tuning. The user confirms this provenance; prediction values alone cannot prove it.

The included example uses a predetermined study and two-soil partition, not an example selected for improvement. Its measured values come from the published study [10.1016/j.chemosphere.2018.10.098](https://doi.org/10.1016/j.chemosphere.2018.10.098). It includes two calibration soils, four test soils, and seven model-prediction columns. Experimental-condition comparability has not been audited for this example, so ranking is off by default.

## Calculations and outputs

The default retains the selected reference model. With zero or one calibration soil it leaves predictions unchanged; with at least two, it fits an intercept and slope with the slope shrunk toward one (penalty 0.1 in squared log-response units). Fits with nonpositive slopes are rejected and replaced by median-offset correction. These counts enable the calculations; they do not guarantee accuracy.

An exploratory mode can select among uploaded models and correction methods using only calibration-soil validation. Its replay did not establish an advantage over fixed calibration, so it is not the default.

The interface reports:

- Mean signed bias (predicted minus measured), including per-soil and per-PFAS results.
- Mean absolute error and the fraction of predictions within twofold on separate test soils.
- Soil-pair ordering error within the same PFAS and comparable conditions.
- Fitted calibration parameters, original and adjusted predictions, and interpretation of the observed results.

Errors give equal weight to soils. Calibration-set fit and test performance are shown separately. Positive-slope calibration preserves soil order; ranking improvement requires changes to the prediction relationships, which this tool does not train. Downloadable CSV, JSON, and Markdown files contain the calculations and adjusted predictions.

## Scope

The tool evaluates existing numerical predictions; it does not require model files or generate Kd from molecular and soil descriptors. It checks input validity and role separation, but cannot certify laboratory measurement quality, site representativeness, or whether a given sample size is sufficient. Censored values and nonpositive raw Kd are not supported.

Evidence is limited to PFAS sorption in soils. The seven-model, 19-study, 180-partition replay checks calculations using existing data; it is not independent external validation of a decision policy. The broader 18-model benchmark does not validate every tool feature on all models. Adapting the software to other endpoints requires revisiting scales, grouping, calibration, and evaluation.

## Tests

```sh
PYTHONPATH=software python -m unittest discover -s software/tests -v
```

Tests cover bias signs, soil weighting, calibration/test separation, preservation of soil order, invalid inputs, and equivalent CSV/XLSX uploads.
