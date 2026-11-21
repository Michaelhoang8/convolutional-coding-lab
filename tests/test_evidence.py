"""Reconcile preserved channel samples, decoding results, plots and aggregates."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import unittest

from convcode.channel import frame
from convcode.core import (
    block_wilson_interval,
    decode_hard,
    decode_soft,
    encode,
    hard_decision,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/published"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def bit_text(values):
    return "".join(str(v) for v in values)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha(path):
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()


class EvidenceTests(unittest.TestCase):
    def test_actual_samples_seeds_codewords_paths_and_error_counts(self):
        rows = load_csv(RUN / "frames.csv")
        raw = (RUN / "received.f64").read_bytes()
        offset = 0
        for row in rows:
            info = tuple(int(v) for v in row["information_bits"])
            length = len(info)
            byte_length = int(row["sample_byte_length"])
            self.assertEqual(int(row["sample_byte_offset"]), offset)
            chunk = raw[offset : offset + byte_length]
            self.assertEqual(hashlib.sha256(chunk).hexdigest(), row["received_sha256"])
            values = struct.unpack("<" + "d" * (byte_length // 8), chunk)
            uncoded, coded = values[:length], values[length:]
            self.assertEqual(len(coded), 2 * (length + 2))
            generated = frame(int(row["snr_index"]), int(row["frame"]), length)
            self.assertEqual(generated["information"], info)
            for key in ["bit_seed", "uncoded_noise_seed", "coded_noise_seed"]:
                self.assertEqual(generated[key], int(row[key]))
            for key in ["uncoded_sigma", "coded_sigma", "effective_rate"]:
                self.assertAlmostEqual(generated[key], float(row[key]), places=14)
            # libm implementations can differ at the last bit in Gaussian draws.
            for a, b in zip(
                values, generated["uncoded_received"] + generated["coded_received"]
            ):
                self.assertAlmostEqual(a, b, places=12)
            encoded = encode(info)
            self.assertEqual(bit_text(encoded.coded), row["encoded_bits"])
            self.assertEqual(bit_text(encoded.states), row["encoded_states"])
            self.assertEqual(
                sum(a != b for a, b in zip(hard_decision(coded), encoded.coded)),
                int(row["coded_channel_sliced_errors"]),
            )
            decoded = {
                "uncoded": hard_decision(uncoded),
                "hard": decode_hard(hard_decision(coded)),
                "soft": decode_soft(coded),
            }
            for name, result in decoded.items():
                bits = result if name == "uncoded" else result.information
                errors = sum(a != b for a, b in zip(info, bits))
                self.assertEqual(bit_text(bits), row[f"{name}_decoded_bits"])
                self.assertEqual(errors, int(row[f"{name}_information_errors"]))
                self.assertEqual(int(errors > 0), int(row[f"{name}_block_error"]))
                if name != "uncoded":
                    self.assertEqual(bit_text(result.states), row[f"{name}_states"])
                    self.assertAlmostEqual(
                        result.metric, float(row[f"{name}_metric"]), places=10
                    )
            offset += byte_length
        self.assertEqual(offset, len(raw))
        self.assertEqual(offset, load_json(RUN / "received_layout.json")["bytes"])

    def test_summary_counts_timing_order_and_frame_intervals_reconcile(self):
        frames = load_csv(RUN / "frames.csv")
        measurements = load_csv(RUN / "measurements.csv")
        environment = load_json(RUN / "environment.json")
        self.assertEqual(len(frames), environment["total_frames"])
        self.assertEqual(len(measurements), 3 * len(frames))
        timing_groups = {}
        for row in measurements:
            timing_groups.setdefault((row["snr_index"], row["frame"]), []).append(row)
            self.assertGreater(int(row["elapsed_ns"]), 0)
        self.assertEqual(len(timing_groups), len(frames))
        for group in timing_groups.values():
            self.assertEqual({r["method"] for r in group}, {"uncoded", "hard", "soft"})
            self.assertEqual({r["order"] for r in group}, {"0", "1", "2"})
        for row in load_csv(RUN / "summary.csv"):
            group = [f for f in frames if f["snr_index"] == row["snr_index"]]
            name = row["method"]
            errors = sum(int(f[f"{name}_information_errors"]) for f in group)
            blocks = sum(int(f[f"{name}_block_error"]) for f in group)
            info_bits = sum(int(f["information_length"]) for f in group)
            low, high = block_wilson_interval(blocks, len(group))
            self.assertEqual(int(row["frames"]), len(group))
            self.assertEqual(int(row["information_bits"]), info_bits)
            self.assertEqual(int(row["information_bit_errors"]), errors)
            self.assertEqual(float(row["information_ber"]), errors / info_bits)
            self.assertEqual(int(row["block_errors"]), blocks)
            self.assertEqual(float(row["block_error_rate"]), blocks / len(group))
            self.assertAlmostEqual(float(row["block_wilson95_low"]), low, places=14)
            self.assertAlmostEqual(float(row["block_wilson95_high"]), high, places=14)
            symbols = sum(
                int(f["information_length" if name == "uncoded" else "coded_length"])
                for f in group
            )
            self.assertEqual(int(row["transmitted_symbols"]), symbols)
            self.assertEqual(float(row["effective_rate"]), info_bits / symbols)
            elapsed = [
                int(t["elapsed_ns"])
                for t in measurements
                if t["snr_index"] == row["snr_index"] and t["method"] == name
            ]
            self.assertEqual(float(row["median_decode_ns"]), statistics.median(elapsed))
            self.assertEqual(int(row["min_decode_ns"]), min(elapsed))
            self.assertEqual(int(row["max_decode_ns"]), max(elapsed))
            if name == "uncoded":
                self.assertAlmostEqual(
                    float(row["uncoded_bpsk_theory_ber"]),
                    0.5
                    * math.erfc(
                        math.sqrt(10 ** (float(row["ebn0_db_information"]) / 10))
                    ),
                    places=14,
                )

    def test_evidence_hashes_trace_and_visual_inputs(self):
        for record in load_json(RUN / "manifest.json"):
            path = RUN / record["path"]
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(sha(path), record["sha256"])
            if path.suffix in (".csv", ".json"):
                self.assertNotIn(b"\r", path.read_bytes())
        for relative, expected in load_json(RUN / "environment.json")[
            "source_sha256_lf"
        ].items():
            self.assertEqual(source_sha(ROOT / relative), expected)
        demo = load_json(RUN / "demo.json")
        for name, result in [
            ("hard", decode_hard(demo["hard_sliced"], trace=True)),
            ("soft", decode_soft(demo["received"], trace=True)),
        ]:
            stored = demo[name]
            self.assertEqual(list(result.states), stored["states"])
            self.assertEqual(list(result.information), stored["information"])
            self.assertEqual(
                [list(v) for v in result.predecessor_history],
                stored["predecessor_history"],
            )
            for actual_stage, saved_stage in zip(
                result.metric_history, stored["metric_history"]
            ):
                for actual, saved in zip(actual_stage, saved_stage):
                    if actual is None or saved is None:
                        self.assertIs(actual, saved)
                    else:
                        self.assertAlmostEqual(actual, saved, places=11)
        assets = ROOT / "docs/assets"
        replay = load_json(assets / "trellis-replay.json")
        self.assertEqual(sha(ROOT / replay["input"]), replay["input_sha256"])
        self.assertEqual(
            source_sha(ROOT / "scripts/render_trellis.py"), replay["source_sha256_lf"]
        )
        self.assertEqual(sha(assets / "trellis-replay.mp4"), replay["video_sha256"])
        self.assertEqual(sha(assets / "trellis.png"), replay["poster_sha256"])
        figures = load_json(assets / "figures.json")
        self.assertEqual(
            source_sha(ROOT / "scripts/build_report.py"), figures["source_sha256_lf"]
        )
        for relative, expected in figures["inputs"].items():
            self.assertEqual(sha(ROOT / relative), expected)
        for relative, expected in figures["outputs"].items():
            self.assertEqual(sha(ROOT / "docs" / relative), expected)


if __name__ == "__main__":
    unittest.main()
