"""Measure terminated convolutional coding on owned seeded AWGN frames."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from convcode.core import (
    block_wilson_interval,
    bpsk,
    decode_hard,
    decode_soft,
    effective_rate,
    encode,
    hard_decision,
    noise_sigma,
)
from convcode.channel import BASE_SEED, EBN0_DB, frame


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bits(values):
    return "".join(str(value) for value in values)


def demo_frame():
    information = (1, 0, 1, 1, 0, 0, 1, 0)
    encoded = encode(information)
    sigma = noise_sigma(2, effective_rate(len(information)))
    # Select a short, explicitly illustrative frame where confidence matters.
    # This separate search is never included in the Monte Carlo aggregate.
    for seed in range(4096):
        rng = random.Random(seed)
        received = tuple(symbol + rng.gauss(0, sigma) for symbol in bpsk(encoded.coded))
        hard = decode_hard(hard_decision(received), trace=True)
        soft = decode_soft(received, trace=True)
        if hard.information != information and soft.information == information:
            return {
                "selection": "First seed in 0..4095 where this fixed 8-bit frame has hard errors and zero soft errors; illustrative and excluded from aggregate",
                "seed": seed,
                "ebn0_db": 2,
                "effective_rate": effective_rate(len(information)),
                "sigma": sigma,
                "information": information,
                "encoded": asdict(encoded),
                "received": received,
                "hard_sliced": hard_decision(received),
                "hard": asdict(hard),
                "soft": asdict(soft),
            }
    raise RuntimeError("No illustration met the predetermined criterion")


def run(output, frames_per_point=400, information_length=64):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Choose a new empty directory; existing evidence is preserved"
        )
    if type(frames_per_point) is not int or not 1 <= frames_per_point <= 2000:
        raise ValueError("Frames per point must lie in 1..2000")
    if type(information_length) is not int or not 1 <= information_length <= 512:
        raise ValueError("Information length must lie in 1..512")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    source_paths = [
        "src/convcode/core.py",
        "src/convcode/channel.py",
        "scripts/run_study.py",
    ]
    hashes = {
        path: hashlib.sha256(
            (ROOT / path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        ).hexdigest()
        for path in source_paths
    }
    order_rng = random.Random(9182)
    records = []
    timings = []
    summary = []
    warm = encode((0, 1) * 32)
    decode_hard(warm.coded)
    decode_soft(bpsk(warm.coded))
    hard_decision(bpsk(warm.information))
    offset = 0
    with (output / "received.f64").open("wb") as samples:
        for snr_index, ebn0 in enumerate(EBN0_DB):
            for index in range(frames_per_point):
                generated = frame(snr_index, index, information_length)
                original = generated["information"]
                encoded = generated["encoded"]
                uncoded_received = generated["uncoded_received"]
                coded_received = generated["coded_received"]
                combined = uncoded_received + coded_received
                raw = struct.pack("<" + "d" * len(combined), *combined)
                samples.write(raw)
                operations = {
                    "uncoded": lambda: hard_decision(uncoded_received),
                    "hard": lambda: decode_hard(hard_decision(coded_received)),
                    "soft": lambda: decode_soft(coded_received),
                }
                names = list(operations)
                order_rng.shuffle(names)
                results = {}
                for order, method in enumerate(names):
                    begin = time.perf_counter_ns()
                    result = operations[method]()
                    elapsed = time.perf_counter_ns() - begin
                    results[method] = result
                    timings.append(
                        {
                            "snr_index": snr_index,
                            "ebn0_db": ebn0,
                            "frame": index,
                            "method": method,
                            "order": order,
                            "elapsed_ns": elapsed,
                        }
                    )
                record = {
                    "snr_index": snr_index,
                    "ebn0_db": ebn0,
                    "frame": index,
                    "information_length": information_length,
                    "coded_length": len(encoded.coded),
                    "effective_rate": generated["effective_rate"],
                    "bit_seed": generated["bit_seed"],
                    "uncoded_noise_seed": generated["uncoded_noise_seed"],
                    "coded_noise_seed": generated["coded_noise_seed"],
                    "uncoded_sigma": generated["uncoded_sigma"],
                    "coded_sigma": generated["coded_sigma"],
                    "sample_byte_offset": offset,
                    "sample_byte_length": len(raw),
                    "received_sha256": hashlib.sha256(raw).hexdigest(),
                    "information_bits": bits(original),
                    "encoded_bits": bits(encoded.coded),
                    "encoded_states": bits(encoded.states),
                    "coded_channel_sliced_errors": sum(
                        a != b
                        for a, b in zip(hard_decision(coded_received), encoded.coded)
                    ),
                }
                offset += len(raw)
                for method, result in results.items():
                    decoded = result if method == "uncoded" else result.information
                    errors = sum(a != b for a, b in zip(decoded, original))
                    record[f"{method}_decoded_bits"] = bits(decoded)
                    record[f"{method}_information_errors"] = errors
                    record[f"{method}_block_error"] = int(errors > 0)
                    if method != "uncoded":
                        record[f"{method}_states"] = bits(result.states)
                        record[f"{method}_metric"] = result.metric
                # Fixed CSV key order independent of timing operation order.
                records.append(dict(sorted(record.items())))
            point = [row for row in records if row["snr_index"] == snr_index]
            for method in ["uncoded", "hard", "soft"]:
                error_bits = sum(row[f"{method}_information_errors"] for row in point)
                error_blocks = sum(row[f"{method}_block_error"] for row in point)
                low, high = block_wilson_interval(error_blocks, frames_per_point)
                elapsed = [
                    row["elapsed_ns"]
                    for row in timings
                    if row["snr_index"] == snr_index and row["method"] == method
                ]
                summary.append(
                    {
                        "snr_index": snr_index,
                        "ebn0_db_information": ebn0,
                        "method": method,
                        "frames": frames_per_point,
                        "information_bits": frames_per_point * information_length,
                        "information_bit_errors": error_bits,
                        "information_ber": error_bits
                        / (frames_per_point * information_length),
                        "block_errors": error_blocks,
                        "block_error_rate": error_blocks / frames_per_point,
                        "block_wilson95_low": low,
                        "block_wilson95_high": high,
                        "effective_rate": (
                            1
                            if method == "uncoded"
                            else effective_rate(information_length)
                        ),
                        "transmitted_symbols": frames_per_point
                        * (
                            information_length
                            if method == "uncoded"
                            else 2 * (information_length + 2)
                        ),
                        "median_decode_ns": statistics.median(elapsed),
                        "min_decode_ns": min(elapsed),
                        "max_decode_ns": max(elapsed),
                        "uncoded_bpsk_theory_ber": (
                            0.5 * math.erfc(math.sqrt(10 ** (ebn0 / 10)))
                            if method == "uncoded"
                            else ""
                        ),
                    }
                )
    write_csv(output / "frames.csv", records)
    write_csv(output / "measurements.csv", timings)
    write_csv(output / "summary.csv", summary)
    write_json(output / "demo.json", demo_frame())
    write_json(
        output / "received_layout.json",
        {
            "format": "IEEE754 float64 little-endian, no header",
            "row_order": "frames.csv order",
            "per_frame": "uncoded_received[information_length], then coded_received[2*(information_length+2)]",
            "bytes": offset,
            "note": "These are actual channel samples passed to the decoders; shared by hard and soft coded comparisons",
        },
    )
    write_json(
        output / "environment.json",
        {
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "os": platform.system(),
            "architecture": platform.machine(),
            "frames_per_point": frames_per_point,
            "information_length": information_length,
            "snr_points_db": EBN0_DB,
            "total_frames": len(records),
            "timed_decoder_calls": len(timings),
            "base_seed": BASE_SEED,
            "order_seed": 9182,
            "source_sha256_lf": hashes,
            "source_hash_policy": "CRLF and CR normalized to LF before hashing, captured before simulation",
            "energy": "Es=1. Eb per information bit includes every transmitted tail symbol. R_eff=L/[2(L+2)] for coded frames, 1 for uncoded.",
            "uncertainty": "Wilson95 intervals only for independent equal-distribution frame-error indicators at fixed Eb/N0; no binomial intervals on correlated decoded bit errors",
            "timing_scope": "One warmup, complete per-frame decoder calls; hard includes slicing; noise generation and file I/O excluded. Each frame is a different observation, not repeated timing of one input.",
            "provenance": "Owned pseudorandom bits and AWGN only; ideal synchronized real BPSK channel, no modem hardware",
        },
    )
    manifest = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "frames": len(records),
                "decoder_calls": len(timings),
                "received_bytes": offset,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/latest")
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--information-bits", type=int, default=64)
    args = parser.parse_args()
    run(args.out, args.frames, args.information_bits)
