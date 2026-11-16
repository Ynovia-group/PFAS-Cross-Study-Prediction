# Data

All numerical tables are UTF-8 CSV files except `study_sources.tsv`, which is tab-separated. They can be read with Excel or `pandas.read_csv`; use `sep="\t"` for the TSV file. Empty cells represent missing values.

## Measured data

| File | Contents | Source |
|---|---|---|
| `soil_records.csv` | 1,274 sorption records from 47 study identifiers; 51 PFAS labels; predictors and response | Fabregat-Palau et al. (2025), [10.1021/acs.est.4c13284](https://doi.org/10.1021/acs.est.4c13284), supplementary workbook `es4c13284_si_002.xlsx` |
| `external_soils.csv` | 151 records for seven PFAS in 23 soils from one separate study | Real et al. (2026), [10.1021/acs.est.5c08377](https://doi.org/10.1021/acs.est.5c08377) |
| `soil_identity.csv` | Mapping of historical rows to 426 aligned soil identifiers | Chen and Vanderborght (2026), [10.1021/acs.est.6c06032](https://doi.org/10.1021/acs.est.6c06032) |
| `response_origins.csv` | Reported/derived response annotations for each historical record | Original compilation and source notes |
| `study_sources.tsv` | Source DOI, record count, and study identifier for the 47 historical studies | Publications underlying the compilation |

A row is a sorption record, not an independent soil. Several PFAS can be measured in one soil, and some publications share soils. The external table is an analysis subset of its source workbook and is separate from historical training. The PFAS counts refer to labels in the source tables, not a new chemical-identity audit.

`row_id` identifies records; `study_id` identifies publications; `chen_soil_id` in the mapping identifies soils. In `study_sources.tsv`, `target_code` is an internal identifier used to match eligible studies to the calibration partitions, not a measured property.

## Response and inputs

The response is `target_log_kd = log10[Kd/(L/kg)]`. Raw Kd and provenance columns are retained for traceability and are not predictors. Use only the allowlisted inputs in `config/features.json`.

| Group | Column | Meaning and unit |
|---|---|---|
| Molecular | `mw_g_mol` | Molecular weight, g/mol |
| Molecular | `pka1`, `pka2` | Acid dissociation constants expressed as pKa |
| Molecular | `log_kow` | Logarithm of the octanol–water partition coefficient |
| Molecular | `carbon_count`, `fluorine_count` | Atom counts |
| Molecular | `pfas_subfamily` | Categorical PFAS subfamily |
| Soil | `ph` | Measured soil pH |
| Soil | `cec_cmol_kg` | Cation exchange capacity, cmol(+)/kg |
| Soil | `corg_percent` | Organic carbon, % |
| Soil | `sand_percent`, `silt_percent`, `clay_percent` | Mineral texture fractions, % |

Author-imputed pH, cation exchange capacity, and texture values were restored to missing where flagged. The packaged training code fits imputation, scaling, and categorical encoding within training folds. Pretrained TabPFN receives the native input columns.

## Partitions and predictions

`config/benchmark_splits.json` stores five publication-grouped outer folds and three inner folds. These reproduce the benchmark partitioning; they are distinct from the soil-identifier exclusions used in calibration experiments.

`config/calibration_splits.json` stores calibration and test rows for 19 eligible studies. Budgets of two, four, and eight measured soils cover 19, 11, and six studies, respectively, with five partitions each: 180 partitions total. For these source predictions, the entire target study and source rows sharing its soil identifiers were excluded from source training. The budget counts soils, not individual PFAS measurements.

| Prediction file | Contents |
|---|---|
| `predictions/benchmark_predictions.csv` | 103,194 predictions of the historical records, including repeated seeds and fixed, tuned, or historical settings |
| `predictions/benchmark_reference.csv` | Reference study-averaged errors and bootstrap intervals |
| `predictions/calibration_predictions.csv` | Seven models' source predictions, consolidated by `target_index`, `model`, and `row_id` |
| `predictions/calibration_reference.csv` | Gaussian process calibration predictions for numerical verification |

Repeated predictions do not add measured data. `fixed` and `tuned` refer to the matched parameter-comparison runs; `historical` retains the pretrained/reference settings as separately labeled results. The calibration replay covers seven models rather than the full 18-model benchmark.

Original supplementary workbooks can be obtained from the linked publishers. Their authors and publishers retain the applicable data rights.
