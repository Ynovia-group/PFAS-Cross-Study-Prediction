"""Scientific invariants and data-isolation tests; synthetic inputs are test-only."""
import base64
import unittest

import numpy as np
import pandas as pd

from diagnostics import InputError, analyze, analyze_study, bias_summary, canonical, fit_calibration, predict, rank_error, soil_errors


def fixture(n=8):
    x = np.arange(n, dtype=float) / 3
    return pd.DataFrame({"study": "S", "soil": [f"s{i}" for i in range(n)], "pfas": "PFOS", "condition": "A",
                         "role": ["update"] * 4 + ["audit"] * (n - 4), "y": x + .7, "pred_A": x, "pred_B": x * 1.2 - .3})


class ScientificTests(unittest.TestCase):
    def test_bias_sign_and_soil_weighting(self):
        f = pd.DataFrame({"soil": ["A"] * 10 + ["B"], "pfas": "PFOS", "y": [0.] * 11})
        b = bias_summary(f, np.array([1.] * 10 + [-3.]))
        self.assertEqual(b["mean_signed_error"], -1.)
        self.assertEqual(b["higher_soils"], 1)
        self.assertEqual(b["lower_soils"], 1)
        self.assertIsNone(b["interval"])

    def test_zero_mean_bias_can_hide_large_errors(self):
        f = pd.DataFrame({"soil": ["A", "B"], "pfas": "PFOS", "y": [0., 0.]})
        prediction = np.array([2., -2.])
        self.assertEqual(bias_summary(f, prediction)["mean_signed_error"], 0.)
        self.assertEqual(soil_errors(f, prediction).mean(), 2.)

    def test_calibration_bias_and_audit_bias_stay_separate(self):
        f = fixture()
        f.loc[f.role.eq("audit"), "y"] -= 2
        r = analyze_study(f, ["pred_A"], "pred_A")
        self.assertAlmostEqual(r["bias"]["calibration_reference"]["mean_signed_error"], -.7)
        self.assertAlmostEqual(r["bias"]["audit_reference"]["mean_signed_error"], 1.3)
        self.assertAlmostEqual(r["bias"]["audit_after"]["mean_signed_error"], 2.)

    def test_bias_requires_observations(self):
        f = fixture().iloc[:0]
        self.assertIsNone(bias_summary(f, np.array([])))

    def test_pfas_biases_are_not_hidden_by_aggregate(self):
        f = pd.DataFrame({"soil": ["A", "B", "A", "B"], "pfas": ["PFOS"] * 2 + ["PFOA"] * 2, "y": [0.] * 4})
        result = bias_summary(f, np.array([1., 1., -1., -1.]))
        self.assertEqual(result["mean_signed_error"], 0.)
        by_pfas = {r["pfas"]: r["mean_signed_error"] for r in result["per_pfas"]}
        self.assertEqual(by_pfas, {"PFOA": -1., "PFOS": 1.})

    def test_constant_shift_is_recovered_without_changing_order(self):
        f = fixture()
        fit = fit_calibration(f.iloc[:4], "pred_A", "affine_shrink")
        self.assertAlmostEqual(fit["slope"], 1)
        self.assertAlmostEqual(fit["intercept"], .7)
        np.testing.assert_allclose(predict(f.iloc[4:], "pred_A", fit), f.iloc[4:].y)
        before = rank_error(f, f.pred_A.to_numpy())
        after = rank_error(f, predict(f, "pred_A", fit))
        self.assertEqual(before, after)

    def test_audit_label_poisoning_cannot_change_selection_or_predictions(self):
        f = fixture()
        a = analyze_study(f, ["pred_A", "pred_B"], "pred_A")
        f.loc[f.role.eq("audit"), "y"] = [100, -50, 3, 18]
        b = analyze_study(f, ["pred_A", "pred_B"], "pred_A")
        self.assertEqual(a["selection"], b["selection"])
        self.assertEqual([r["adjusted_log10_kd"] for r in a["export"]], [r["adjusted_log10_kd"] for r in b["export"]])
        self.assertNotEqual(a["audit"]["after"]["mae"], b["audit"]["after"]["mae"])

    def test_soil_not_record_weighting(self):
        f = pd.DataFrame({"soil": ["A"] * 10 + ["B"], "y": [0.] * 11})
        self.assertEqual(soil_errors(f, np.array([1.] * 10 + [3.])).mean(), 2)

    def test_no_cross_compound_ranking_pairs(self):
        f = fixture(8)
        f["pfas"] = ["A"] * 4 + ["B"] * 4
        r = rank_error(f, f.pred_A.to_numpy())
        self.assertEqual(r["pairs"], 12)  # 6 pairs per compound, not 28 across all.
        f["condition"] = ["X", "X", "Y", "Y"] * 2
        self.assertEqual(rank_error(f, f.pred_A.to_numpy())["pairs"], 4)

    def test_negative_affine_slope_excluded(self):
        f = fixture()
        f["y"] = -f.pred_A * 20
        f["role"] = ["update"] * 2 + ["audit"] * 6
        r = analyze_study(f, ["pred_A"], "pred_A")
        self.assertEqual(r["selection"]["selected"]["method"], "offset_median")
        self.assertGreater(len(r["selection"]["rejected"]), 0)

    def test_two_soils_dont_select_model_using_target_response(self):
        f = fixture()
        f["role"] = ["update"] * 2 + ["audit"] * 6
        f["pred_B"] = f.y  # A tempting perfect alternative must not trigger a model switch.
        r = analyze_study(f, ["pred_A", "pred_B"], "pred_A")
        self.assertEqual(r["selection"]["selected"]["model"], "pred_A")

    def test_zero_and_one_measured_soil_preserve_source(self):
        for n in [0, 1]:
            f = fixture()
            f["role"] = ["update"] * n + ["predict"] * (8 - n)
            f.loc[f.role.eq("predict"), "y"] = np.nan
            r = analyze_study(f, ["pred_A"], "pred_A")
            self.assertEqual(r["selection"]["selected"]["method"], "source")
            self.assertIsNone(r["audit"])

    def test_prediction_ties_count_half_and_observed_ties_excluded(self):
        f = fixture().iloc[:3].copy()
        f["y"] = [0, 0, 1]
        r = rank_error(f, np.zeros(3))
        self.assertEqual(r["pairs"], 2)
        self.assertEqual(r["error"], .5)


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {"soil": "soil_id", "pfas": "pfas", "observed": "observed", "role": "role", "predictions": ["pred_A"]}
        self.options = {"source_independent": True, "observed_scale": "log10", "prediction_scale": "log10"}
        self.frame = pd.DataFrame({"soil_id": ["A", "B", "C"], "pfas": "PFOS", "observed": ["1", "2", ""], "pred_A": [".5", "1.5", "2.5"], "role": ["update", "audit", "predict"]})

    def test_same_soil_split_between_roles_rejected(self):
        self.frame.loc[1, "soil_id"] = "A"
        with self.assertRaises(InputError):
            canonical(self.frame, self.mapping, self.options)

    def test_censored_or_nonpositive_raw_kd_rejected(self):
        for value, scale in [("<0.1", "log10"), ("0", "kd"), ("-1", "kd"), ("inf", "log10")]:
            self.frame.loc[0, "observed"] = value
            with self.assertRaises(InputError):
                canonical(self.frame, self.mapping, {**self.options, "observed_scale": scale})

    def test_raw_and_log_scale_are_equivalent(self):
        a, _ = canonical(self.frame, self.mapping, self.options)
        self.frame["pred_A"] = (10 ** self.frame.pred_A.astype(float)).astype(str)
        b, _ = canonical(self.frame, self.mapping, {**self.options, "prediction_scale": "kd"})
        np.testing.assert_allclose(a.pred_A, b.pred_A)

    def test_hidden_labels_in_predict_rows_rejected(self):
        self.frame.loc[2, "observed"] = "5"
        with self.assertRaises(InputError):
            canonical(self.frame, self.mapping, self.options)

    def test_observed_cannot_be_prediction(self):
        with self.assertRaises(InputError):
            canonical(self.frame, {**self.mapping, "predictions": ["observed"]}, self.options)

    def test_reserved_prediction_column_cannot_overwrite_soil_identifier(self):
        self.frame["soil"] = "0.3"
        with self.assertRaises(InputError):
            canonical(self.frame, {**self.mapping, "predictions": ["soil"]}, self.options)

    def test_excel_and_csv_uploads_produce_same_diagnosis(self):
        import io
        buffer = io.BytesIO()
        self.frame.to_excel(buffer, index=False)
        common = {"mapping": self.mapping, "options": self.options}
        excel = analyze({**common, "filename": "data.xlsx", "data": base64.b64encode(buffer.getvalue()).decode()})
        csv = analyze({**common, "filename": "data.csv", "data": base64.b64encode(self.frame.to_csv(index=False).encode()).decode()})
        self.assertEqual(excel, csv)

    def test_end_to_end_missing_observations_allowed(self):
        text = self.frame.to_csv(index=False).encode()
        payload = {"data": base64.b64encode(text).decode(), "filename": "test.csv", "mapping": self.mapping, "options": self.options}
        result = analyze(payload)
        import json
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["studies"][0]["counts"]["predict_soils"], 1)

    def test_multiple_studies_are_fitted_separately(self):
        a = fixture()
        b = fixture()
        b["study"] = "T"
        b["y"] += 20
        ra = analyze_study(a, ["pred_A"], "pred_A")
        rb = analyze_study(b, ["pred_A"], "pred_A")
        self.assertNotEqual(ra["selection"]["selected"]["params"], rb["selection"]["selected"]["params"])


if __name__ == "__main__":
    unittest.main()
