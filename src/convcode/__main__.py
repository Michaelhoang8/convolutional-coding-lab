"""Encode binary text or decode JSON observations with a terminated Viterbi path."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .core import encode, decode_hard, decode_soft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    encoder = sub.add_parser("encode")
    encoder.add_argument("bits", help="Binary information-bit string")
    for mode in ["hard", "soft"]:
        decoder = sub.add_parser(mode)
        decoder.add_argument(
            "input", type=Path, help="JSON array or object with received array"
        )
        decoder.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    try:
        if args.operation == "encode":
            if any(bit not in "01" for bit in args.bits):
                raise ValueError("Input must contain only 0 and 1")
            result = encode(tuple(int(bit) for bit in args.bits))
        else:
            value = json.loads(args.input.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value = value["received"]
            result = (decode_hard if args.operation == "hard" else decode_soft)(
                value, trace=args.trace
            )
        print(json.dumps(asdict(result), indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
