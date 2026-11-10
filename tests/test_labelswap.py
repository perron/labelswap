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

    def test_module_entry_point_runs(self):
        r = subprocess.run([sys.executable, "-m", "labelswap", "--version"], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0)
        self.assertIn("labelswap", r.stdout)


if __name__ == "__main__":
    unittest.main()
