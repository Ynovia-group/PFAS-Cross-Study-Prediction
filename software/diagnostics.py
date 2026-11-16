"""PFAS prediction diagnostics. No fitting or selection may read audit responses."""
from __future__ import annotations

import base64
import io
import math

import numpy as np
import pandas as pd

VERSION = "0.1.0"
METHODS = ("source", "offset_median", "affine_shrink")
METHOD_NAMES = {"source": "Original predictions", "offset_median": "Median offset correction", "affine_shrink": "Regularized calibration"}
LAMBDA = 0.1


class InputError(ValueError):
    pass


def read_upload(payload):
    try:
        raw = base64.b64decode(payload["data"], validate=True)
        if len(raw) > 8_000_000:
            raise InputError("The file exceeds 8 MB. Split it by study before uploading.")
        name = payload.get("filename", "data.csv").lower()
        if name.endswith(".xlsx"):
            frame = pd.read_excel(io.BytesIO(raw), dtype=str).fillna("")
        elif name.endswith((".csv", ".tsv")):
            text = raw.decode("utf-8-sig")
            sep = "\t" if name.endswith(".tsv") else ","
            import csv
            header = next(csv.reader(io.StringIO(text), delimiter=sep))
            if len(header) != len(set(header)):
                raise InputError("Duplicate column names found. Give each column a unique name.")
            frame = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
        else:
            raise InputError("Upload a UTF-8 CSV or TSV file, or an XLSX workbook.")
    except InputError:
        raise
    except Exception as exc:
        raise InputError("Unable to read the file. Check its format, encoding, and column headers.") from exc
    if not 1 <= len(frame) <= 10000:
        raise InputError("The file must contain 1–10,000 data rows.")
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.columns.duplicated().any() or any(not c for c in frame.columns):
        raise InputError("Column names must be nonempty and unique.")
    return frame


def inspect(payload):
    frame = read_upload(payload)
    return {"columns": frame.columns.tolist(), "rows": len(frame), "preview": frame.head(5).to_dict("records")}


def canonical(frame, mapping, options):
    models = mapping.get("predictions", [])
    if not 1 <= len(models) <= 18 or len(set(models)) != len(models):
        raise InputError("Select 1–18 distinct prediction columns.")
    if set(models) & {"study", "soil", "pfas", "condition", "role", "row", "y"}:
        raise InputError("A prediction column conflicts with an internal identifier. Rename it with a pred_ prefix, for example pred_model_A.")
    for key, col in mapping.items():
        if key != "predictions" and col and col not in frame:
            raise InputError(f"Column not found: {col}")
    for column in models + [mapping.get("soil"), mapping.get("pfas")]:
        if not column or column not in frame:
            raise InputError("Specify a soil identifier, a PFAS name column, and at least one prediction column.")
    if mapping.get("observed") in models:
        raise InputError("The measured-value column cannot also be a prediction column.")
    out = pd.DataFrame(index=frame.index)
    for key, default in [("study", "Uploaded study"), ("soil", ""), ("pfas", ""), ("condition", "Unspecified common conditions")]:
        col = mapping.get(key)
        out[key] = frame[col].astype(str).str.strip() if col else default
        if out[key].eq("").any():
            raise InputError(f"The {key} column contains empty values. Complete the identifiers or names.")
    out["row"] = np.arange(len(frame)) + 2
    obs = mapping.get("observed")
    original = frame[obs].astype(str).str.strip() if obs else pd.Series("", index=frame.index)
    out["y"] = numeric(original, "Measured values", allow_missing=True)
    for model in models:
        out[model] = numeric(frame[model], model)
    for col, scale in [("y", options.get("observed_scale", "log10"))] + [(m, options.get("prediction_scale", "log10")) for m in models]:
        if scale not in ("log10", "kd"):
            raise InputError("Select either Kd in L/kg or log10[Kd/(L/kg)].")
        if scale == "kd":
            if (out[col].dropna() <= 0).any():
                raise InputError(f"Column {col} contains nonpositive Kd values, which cannot be log-transformed.")
            out[col] = np.log10(out[col])
    role = mapping.get("role")
    if role:
        out["role"] = frame[role].astype(str).str.strip().str.lower()
    else:
        out["role"] = np.where(out.y.notna(), "update", "predict")
    if not set(out.role) <= {"update", "audit", "predict"}:
        raise InputError("The role column must contain update, audit, or predict.")
    if out.loc[out.role.isin(["update", "audit"]), "y"].isna().any():
        raise InputError("Every update or audit row must contain a measured value.")
    if out.loc[out.role.eq("predict"), "y"].notna().any():
        raise InputError("A predict row cannot contain a measured value. Assign it to update or audit instead.")
    if out.groupby(["study", "soil"]).role.nunique().gt(1).any():
        raise InputError("The same soil has different roles within a study. Assign all its records to the same role.")
    counts = out[out.role.eq("update")].groupby("study").soil.nunique()
    if len(counts) and counts.max() > 200:
        raise InputError("This interface supports up to 200 calibration soils per study. Use a batch workflow for larger datasets.")
    if not options.get("source_independent"):
        raise InputError("Confirm that the source models were trained and tuned without measured Kd from the target study.")
    return out, models


def numeric(series, name, allow_missing=False):
    s = series.astype(str).str.strip()
    missing = s.isin(["", "NA", "NaN", "nan", "N/A", "null"])
    n = pd.to_numeric(s.where(~missing, np.nan), errors="coerce")
    invalid = (~missing & n.isna()) | (n.notna() & ~np.isfinite(n))
    if invalid.any() or (missing.any() and not allow_missing):
        raise InputError(f"Column {name} contains invalid numbers, detection-limit symbols, or missing predictions. Check the file.")
    return n.astype(float)


def fit_calibration(known, model, method):
    if method == "source":
        return {"intercept": 0.0, "slope": 1.0}
    if known.empty:
        return None
    z, y = known[model].to_numpy(float), known.y.to_numpy(float)
    if method == "offset_median":
        residuals = pd.DataFrame({"soil": known.soil.to_numpy(), "r": y - z})
        offset = float(residuals.groupby("soil").r.median().median())
        return {"intercept": offset, "slope": 1.0}
    w = 1 / known.groupby("soil").soil.transform("size").to_numpy(float)
    w = w / w.sum()
    zm, ym = float(w @ z), float(w @ y)
    variance = float(w @ ((z - zm) ** 2))
    covariance = float(w @ ((z - zm) * (y - ym)))
    slope = (covariance + LAMBDA) / (variance + LAMBDA)
    return {"intercept": ym - slope * zm, "slope": slope, "lambda": LAMBDA}


def predict(frame, model, params):
    return params["intercept"] + params["slope"] * frame[model].to_numpy(float)


def soil_errors(frame, predictions):
    d = pd.DataFrame({"soil": frame.soil.to_numpy(), "ae": abs(predictions - frame.y.to_numpy(float))})
    return d.groupby("soil", sort=True).ae.mean()


def choose(known, models, reference, policy="exploratory_cv"):
    """Only update rows enter this function. Audit values are never supplied."""
    ordered_models = [reference] + [m for m in models if m != reference]
    n = known.soil.nunique()
    candidates, rejected = [], []
    for method in METHODS:
        for model in ordered_models:
            params = fit_calibration(known, model, method)
            if params is None:
                continue
            if params["slope"] <= 0:
                rejected.append({"model": model, "method": method, "slope": params["slope"], "reason": "The fitted slope is nonpositive and could reverse or collapse the soil order."})
                continue
            losses = []
            if n >= 2:
                for held in sorted(known.soil.unique()):
                    train, valid = known[known.soil.ne(held)], known[known.soil.eq(held)]
                    fitted = fit_calibration(train, model, method)
                    if fitted is None or fitted["slope"] <= 0:
                        losses = []
                        break
                    losses.append(float(np.mean(abs(predict(valid, model, fitted) - valid.y.to_numpy()))))
            candidates.append({"model": model, "method": method, "method_label": METHOD_NAMES[method], "params": params,
                               "validation_mae": float(np.mean(losses)) if losses else None,
                               "validation_se": float(np.std(losses, ddof=1) / math.sqrt(len(losses))) if len(losses) >= 2 else None,
                               "validation_soils": len(losses), "fold_errors": losses})
    if n < 2:
        selected = next(c for c in candidates if c["model"] == reference and c["method"] == "source")
        reason = "Fewer than two soils are available for calibration, so the specified reference model is retained. These data cannot establish calibration performance on other soils."
    elif n == 2 or policy == "fixed_calibration":
        selected = next((c for c in candidates if c["model"] == reference and c["method"] == "affine_shrink"), None)
        if selected is None:
            selected = next(c for c in candidates if c["model"] == reference and c["method"] == "offset_median")
        reason = "The reference model is calibrated using the predefined procedure. Validation errors are shown for comparison but do not trigger a model switch. Performance must be assessed on separate test soils."
        if n == 2:
            reason = "Two measured soils are available. " + reason
    else:
        valid = [c for c in candidates if c["validation_mae"] is not None]
        best = min(valid, key=lambda c: c["validation_mae"])
        threshold = best["validation_mae"] + best["validation_se"]
        selected = next(c for c in valid if c["validation_mae"] <= threshold + 1e-12)
        reason = "Leave-one-soil-out validation uses only update soils. The first candidate within one standard error of the minimum error is selected, with priority given to simpler procedures. These validation errors guide selection; separate test soils are needed to assess performance."
    return {"selected": selected, "candidates": candidates, "rejected": rejected, "reason": reason, "update_soils": int(n), "policy": policy}


def rank_error(frame, predictions):
    """A single fitted mapping, within compound AND condition; no cross-fold ranks."""
    d = frame[["soil", "pfas", "condition", "y"]].copy()
    d["p"] = predictions
    d = d.groupby(["pfas", "condition", "soil"], as_index=False)[["y", "p"]].mean()
    per_stratum = []
    errors, pairs = 0.0, 0
    for (pfas, condition), group in d.groupby(["pfas", "condition"]):
        y, p = group.y.to_numpy(), group.p.to_numpy()
        group_errors, group_pairs = 0.0, 0
        # Linear memory, even when many audit soils are uploaded.
        for i in range(len(y) - 1):
            dy, dp = y[i] - y[i + 1:], p[i] - p[i + 1:]
            use = np.abs(dy) > 1e-12
            err = np.where(np.abs(dp[use]) <= 1e-12, .5, (dy[use] * dp[use] < 0).astype(float))
            group_errors += float(err.sum())
            group_pairs += len(err)
        if not group_pairs:
            continue
        per_stratum.append({"pfas": pfas, "condition": condition, "pairs": group_pairs, "error": group_errors / group_pairs})
        pairs += group_pairs
        errors += group_errors
    return {"error": errors / pairs if pairs else None, "pairs": int(pairs), "strata": per_stratum,
            "aggregation": "Pool comparable soil pairs within the same PFAS and condition; exclude measured ties and count predicted ties as half an error."}


def metrics(frame, predictions, ranking=False):
    errors = soil_errors(frame, predictions)
    r = pd.DataFrame({"soil": frame.soil.to_numpy(), "r": predictions - frame.y.to_numpy(float)})
    return {"mae": float(errors.mean()), "signed_bias": float(r.groupby("soil").r.mean().mean()),
            "within_twofold": float(r.assign(ok=abs(r.r) <= np.log10(2)).groupby("soil").ok.mean().mean()),
            "soils": len(errors), "records": len(frame), "soil_errors": errors.to_dict(),
            "ranking": rank_error(frame, predictions) if ranking else None}


def bias_summary(frame, predictions, bootstrap=2000):
    """Empirical signed error; descriptive only, never used for method selection."""
    if frame.empty:
        return None
    residuals = frame[["soil", "pfas"]].copy()
    residuals["error"] = np.asarray(predictions, dtype=float) - frame.y.to_numpy(float)
    per_soil = residuals.groupby("soil", sort=True).error.mean()
    values = per_soil.to_numpy()
    estimate = float(values.mean())
    interval = None
    if len(values) >= 4 and bootstrap:
        rng = np.random.default_rng(20261115)
        # Limit memory while retaining soil-level resampling.
        samples = []
        for start in range(0, bootstrap, 100):
            indices = rng.integers(len(values), size=(min(100, bootstrap - start), len(values)))
            samples.extend(values[indices].mean(axis=1).tolist())
        interval = np.quantile(samples, [.025, .975]).tolist()
    per_pfas = []
    for pfas, group in residuals.groupby("pfas", sort=True):
        soil_means = group.groupby("soil").error.mean()
        per_pfas.append({"pfas": pfas, "soils": len(soil_means), "records": len(group),
                         "mean_signed_error": float(soil_means.mean())})
    return {"mean_signed_error": estimate, "interval": interval, "soils": len(per_soil), "records": len(frame),
            "higher_soils": int((values > 1e-12).sum()), "lower_soils": int((values < -1e-12).sum()),
            "per_soil": [{"soil": s, "mean_signed_error": float(v)} for s, v in per_soil.items()],
            "per_pfas": per_pfas}


def gain(before, after, bootstrap=2000):
    a, b = pd.Series(before["soil_errors"]), pd.Series(after["soil_errors"])
    delta = (a - b).to_numpy()
    mean = float(delta.mean())
    ci = None
    if len(delta) >= 4 and bootstrap:
        rng = np.random.default_rng(20261115)
        draws = delta[rng.integers(len(delta), size=(bootstrap, len(delta)))].mean(axis=1)
        ci = np.quantile(draws, [.025, .975]).tolist()
    return {"absolute": mean, "percent": 100 * mean / before["mae"] if before["mae"] > 1e-12 else None,
            "interval": ci, "improved_soils": int((delta > 1e-12).sum()), "worsened_soils": int((delta < -1e-12).sum()),
            "unchanged_soils": int((abs(delta) <= 1e-12).sum()),
            "per_soil": [{"soil": s, "gain": float(v)} for s, v in (a - b).items()]}


def analyze_study(frame, models, reference, ranking=False, bootstrap=2000, policy="fixed_calibration"):
    known = frame[frame.role.eq("update")].copy()
    selection = choose(known, models, reference, policy)
    selected = selection["selected"]
    model, params = selected["model"], selected["params"]
    # Everything used to choose/fix the method is complete before audit.y is read.
    fitted_prediction = predict(frame, model, params)
    audit = frame[frame.role.eq("audit")].copy()
    audit_result = None
    warnings = []
    if policy == "exploratory_cv":
        warnings.append("Exploratory automatic selection did not outperform fixed calibration in the replay of the first study. The selected procedure is not an established optimum.")
    if not audit.empty:
        before = metrics(audit, audit[reference].to_numpy(), ranking)
        chosen_before = metrics(audit, audit[model].to_numpy(), ranking)
        after = metrics(audit, predict(audit, model, params), ranking)
        total_gain = gain(before, after, bootstrap)
        calibration_gain = gain(chosen_before, after, bootstrap)
        audit_result = {"reference_before": before, "chosen_model_before": chosen_before, "after": after,
                        "total_gain": total_gain, "calibration_gain": calibration_gain,
                        "points": [{"soil": row.soil, "pfas": row.pfas, "condition": row.condition, "observed": float(row.y),
                                    "before": float(row[reference]), "after": float(p)}
                                   for (_, row), p in zip(audit.iterrows(), predict(audit, model, params))]}
        interval = total_gain["interval"]
        if total_gain["absolute"] < -1e-12:
            verdict = "Error increased on the separate test soils"
            action = "The selected procedure did not improve numerical predictions on these test soils. Check whether the calibration soils represent the soils to be predicted, and evaluate any revised procedure on new test data. Repeated selection on the same test set would compromise its independence."
        elif total_gain["absolute"] > 1e-12:
            verdict = "Error decreased on the separate test soils"
            action = "The adjusted predictions are available for download. Their usefulness for other soils depends on how well those soils are represented by the calibration and test data."
        else:
            verdict = "The selected procedure retained the original predictions"
            action = "The selection rule did not apply an additional correction. Check the absolute test error to assess whether the original predictions meet your accuracy requirements."
        if interval and interval[0] <= 0 <= interval[1] and abs(total_gain["absolute"]) > 1e-12:
            action += " The descriptive interval includes zero, so these soils do not establish a consistent direction of improvement."
        if len(audit.soil.unique()) < 4:
            warnings.append("Fewer than four test soils are available. Observed differences are reported without bootstrap intervals.")
    else:
        verdict = "Candidate predictions are available; separate testing is still needed"
        action = "Reserve separate measured soils to assess whether the selected procedure improves predictions for other soils. Internal validation guides method selection and does not replace this test."
        if known.empty:
            verdict = "Prediction error in the target study cannot yet be assessed"
            action = "The reference predictions are retained. Representative measured Kd values from the target study are needed for evaluation or calibration."
    if 0 < known.soil.nunique() <= 2:
        warnings.append("The available soil count does not guarantee accuracy. Measuring several PFAS in one soil still provides one soil, not several independent soils.")
    unseen = sorted(set(frame.loc[frame.role.ne("update"), "pfas"]) - set(known.pfas))
    if len(known) and unseen:
        warnings.append("The following PFAS were absent from the calibration data; applying the same correction to them requires further evaluation: " + ", ".join(unseen[:12]))
    if selection["rejected"]:
        warnings.append("Affine fits with nonpositive slopes were excluded. See the parameter details.")
    if not ranking:
        warnings.append("Comparable experimental conditions have not been confirmed, so soil-ranking error was not calculated.")
    ranking_message = "Offset correction and calibration with a positive slope preserve each model's soil order. Lower numerical error does not imply better ranking."
    if model != reference:
        ranking_message += " A different model was also selected. Any change in soil order comes from model selection, not from calibration."
    bias = {
        "definition": "Mean prediction bias is predicted minus measured log10 Kd, averaged first within each soil and then equally across soils. Positive values indicate overprediction; negative values indicate underprediction.",
        "calibration_reference": bias_summary(known, known[reference].to_numpy(float), bootstrap),
        "audit_reference": bias_summary(audit, audit[reference].to_numpy(float), bootstrap),
        "audit_after": bias_summary(audit, predict(audit, model, params), bootstrap),
        "interpretation": "This describes average bias in the measured soils. Opposing errors can cancel, so also inspect absolute error and individual soils. It neither identifies the cause of bias nor establishes a common bias across all soils or PFAS.",
    }
    export = []
    for (_, row), p in zip(frame.iterrows(), fitted_prediction):
        export.append({"study_id": row.study, "soil_id": row.soil, "pfas": row.pfas, "condition_id": row.condition,
                       "role": row.role, "observed_log10_kd": float(row.y) if pd.notna(row.y) else None,
                       "reference_model": reference, "reference_log10_kd": float(row[reference]),
                       "selected_model": model, "method": selected["method"], "selected_source_log10_kd": float(row[model]),
                       "adjusted_log10_kd": float(p), "adjusted_kd_l_kg": float(10.0 ** p) if -300 < p < 300 else None})
    return {"study": str(frame.study.iloc[0]), "counts": {"records": len(frame), "soils": frame.soil.nunique(),
            "update_records": len(known), "update_soils": known.soil.nunique(), "audit_records": len(audit),
            "audit_soils": audit.soil.nunique(), "predict_soils": frame[frame.role.eq("predict")].soil.nunique()},
            "selection": selection, "audit": audit_result, "bias": bias, "verdict": verdict, "action": action,
            "warnings": warnings, "ranking_message": ranking_message, "export": export}


def analyze(payload):
    frame = read_upload(payload)
    mapping, options = payload.get("mapping", {}), payload.get("options", {})
    data, models = canonical(frame, mapping, options)
    reference = options.get("reference_model", models[0])
    if reference not in models:
        raise InputError("The reference model must be one of the selected prediction columns.")
    policy = options.get("policy", "fixed_calibration")
    if policy not in ("fixed_calibration", "exploratory_cv"):
        raise InputError("Unknown analysis mode.")
    studies = [analyze_study(g, models, reference, bool(options.get("ranking_comparable")), policy=policy) for _, g in data.groupby("study", sort=False)]
    missing = []
    for column in frame.columns:
        if column.lower() in {"corg_percent", "organic_carbon", "organic_carbon_percent", "ph", "cec_cmol_kg", "mw_g_mol"}:
            blank = frame[column].astype(str).str.strip().isin(["", "NA", "NaN", "nan", "N/A"])
            missing.append({"column": column, "missing": int(blank.sum()), "records": len(frame)})
    return {"version": VERSION, "reference_model": reference, "models": models, "studies": studies,
            "input_missingness": missing, "provenance": "The user confirms that source predictions are independent of the target measurements. A CSV alone cannot verify training provenance.",
            "metric_scale": "Errors are evaluated in log10[Kd/(L/kg)]; absolute errors are averaged first within each soil and then equally across soils.",
            "scope": "This tool compares uploaded predictions and calibration procedures. It does not train models that predict Kd from soil and molecular descriptors."}
