import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from labelswap.cli import main
from labelswap.model import Model, in_holdout, shadow, tokenize, tune

ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    ("refund my double charge", "billing"),
    ("invoice charge was wrong", "billing"),
    ("card declined on payment", "billing"),
    ("site is down 503", "outage"),
    ("dashboard down and timing out", "outage"),
    ("latency alarm fired, service down", "outage"),
]


class ModelTests(unittest.TestCase):
    def test_tokenize_adds_bigrams(self):
        self.assertEqual(tokenize("Site DOWN now"), ["site", "down", "now", "site_down", "down_now"])

    def test_tokenize_keeps_unicode_words(self):
        self.assertEqual(
            tokenize("Réclamation overdue"),
            ["réclamation", "overdue", "réclamation_overdue"],
        )
        self.assertIn("münchen", tokenize("München billing"))
        self.assertEqual(tokenize("结算失败"), ["结算失败"])

    def test_predicts_obvious_labels(self):
        m = Model().fit(ROWS)
        self.assertEqual(m.decide("my charge needs a refund")["label"], "billing")
        self.assertEqual(m.decide("the service is down")["label"], "outage")

    def test_probabilities_sum_to_one(self):
        p = Model().fit(ROWS).proba("anything at all")
        self.assertAlmostEqual(sum(p.values()), 1.0)

    def test_untuned_model_defers_everything(self):
        self.assertEqual(Model().fit(ROWS).decide("site is down")["action"], "defer")

    def test_tune_meets_target_on_tuning_rows(self):
        m = Model().fit(ROWS)
        rep = tune(m, ROWS, 1.0)
        self.assertTrue(rep["met"])
        self.assertEqual(rep["agreement_on_local"], 1.0)
        self.assertGreater(rep["coverage"], 0)

    def test_impossible_target_defers_everything(self):
        m = Model().fit(ROWS)
        wrong = [(t, "billing" if l == "outage" else "outage") for t, l in ROWS]
        rep = tune(m, wrong, 0.9)
        self.assertFalse(rep["met"])
        self.assertEqual(rep["local_calls"], 0)

    def test_round_trip_keeps_decisions(self):
        m = Model().fit(ROWS)
        tune(m, ROWS, 1.0)
        m2 = Model.from_json(json.loads(json.dumps(m.to_json())))
        for text, _ in ROWS:
            self.assertEqual(m.decide(text), m2.decide(text))

    def test_rejects_foreign_model_file(self):
        with self.assertRaises(ValueError):
            Model.from_json({"format": "other"})

    def test_holdout_is_deterministic(self):
        self.assertEqual(in_holdout("abc", 0.5), in_holdout("abc", 0.5))
        self.assertFalse(in_holdout("abc", 0))

    def test_shadow_counts_mistakes(self):
        m = Model(threshold=0.0).fit(ROWS)
        rep = shadow(m, [("site is down", "billing")])
        self.assertEqual(rep["local_calls"], 1)
        self.assertEqual(rep["top_mistakes"][0]["reference"], "billing")

    def test_gate_and_shadow_use_unrounded_confidence(self):
        """tune used to store round(conf, 6) as the gate, so an example that
        set the cut could then defer because its raw score was just below the
        rounded-up threshold. shadow also compared the rounded display value.
        """
        m = Model().fit([
            ("alpha beta gamma", "L1"),
            ("alpha beta gamma delta", "L1"),
            ("omega zeta eta", "L2"),
            ("omega zeta", "L2"),
        ])
        text = "omega foo zeta delta"
        label, raw = m.top(text)
        self.assertEqual(label, "L2")
        self.assertGreater(round(raw, 6), raw)
        m.threshold = raw
        self.assertEqual(m.decide(text)["action"], "local")
        self.assertEqual(shadow(m, [(text, "L2")])["local_calls"], 1)
        # shadow must follow decide even if the gate sits on the rounded display value
        m.threshold = round(raw, 6)
        action = m.decide(text)["action"]
        rep = shadow(m, [(text, "L2")])
        if action == "local":
            self.assertEqual(rep["local_calls"], 1)
        else:
            self.assertEqual(rep["deferred_calls"], 1)

    def test_tuned_gate_accepts_the_cut_example(self):
        m = Model().fit(ROWS)
        rows = ROWS + [("site is down 503", "outage"), ("card charge refund", "billing")]
        tune(m, rows, 0.5)
        accepted = 0
        for text, _ref in rows:
            _label, raw = m.top(text)
            d = m.decide(text)
            if raw >= m.threshold:
                self.assertEqual(d["action"], "local")
                accepted += 1
            else:
                self.assertEqual(d["action"], "defer")
        self.assertEqual(shadow(m, rows)["local_calls"], accepted)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "t.jsonl"
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_tickets.py"), str(self.data)], check=True, cwd=ROOT)

    def run_cli(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(args))
        return code, buf.getvalue()

    def test_train_shadow_decide_end_to_end(self):
        model = self.tmp / "m.json"
        code, out = self.run_cli("train", str(self.data), "-o", str(model), "--target", "0.98", "--json")
        self.assertEqual(code, 0)
        rep = json.loads(out)["report"]
        self.assertTrue(rep["met"])
        self.assertGreaterEqual(rep["agreement_on_local"], 0.98)
        self.assertGreater(rep["coverage"], 0.5)
        code, out = self.run_cli("shadow", str(model), str(self.data), "--json")
        self.assertGreater(json.loads(out)["agreement_on_local"], 0.9)
        code, out = self.run_cli("decide", str(model), "--text", "checkout is down and returns 503 errors")
        self.assertEqual(json.loads(out)["label"], "outage")

    def test_bad_json_line_reports_line_number(self):
        bad = self.tmp / "bad.jsonl"
        bad.write_text('{"text": "a", "label": "b"}\nnot json\n')
        with self.assertRaises(SystemExit) as cm:
            self.run_cli("train", str(bad))
        self.assertIn(":2:", str(cm.exception))

    def test_non_object_jsonl_reports_line_number(self):
        bad = self.tmp / "bad.jsonl"
        bad.write_text("123\n")
        with self.assertRaises(SystemExit) as cm:
            self.run_cli("train", str(bad))
        msg = str(cm.exception)
        self.assertIn(":1:", msg)
        self.assertIn("JSON object", msg)

    def test_missing_model_file_is_clean_error(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_cli("decide", str(self.tmp / "nope.json"), "--text", "x")
        self.assertIn("not found", str(cm.exception).lower())

    def test_foreign_model_file_is_clean_error(self):
        path = self.tmp / "foreign.json"
        path.write_text('{"format": "other"}')
        with self.assertRaises(SystemExit) as cm:
            self.run_cli("decide", str(path), "--text", "x")
        self.assertIn("cannot load model", str(cm.exception))

    def test_target_out_of_range_is_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_cli("train", str(self.data), "--target", "1.5")
        self.assertEqual(cm.exception.code, 2)

    def test_module_entry_point_runs(self):
        r = subprocess.run([sys.executable, "-m", "labelswap", "--version"], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0)
        self.assertIn("labelswap", r.stdout)


if __name__ == "__main__":
    unittest.main()
