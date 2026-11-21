"""Hand vectors and exhaustive independent terminated-codeword ML oracles."""

import itertools
import math
import random
import unittest

from convcode.core import (
    block_wilson_interval,
    bpsk,
    decode_hard,
    decode_soft,
    effective_rate,
    encode,
    hard_decision,
    noise_sigma,
    transition,
)


def reference_codeword(information):
    # Polynomial dot products over a three-bit register, independent of the
    # production state numbering and transition table.
    register = [0, 0, 0]
    output = []
    for value in tuple(information) + (0, 0):
        register = [value, register[0], register[1]]
        for taps in [(1, 1, 1), (1, 0, 1)]:
            output.append(sum(a * b for a, b in zip(register, taps)) % 2)
    return tuple(output)


class CodecTests(unittest.TestCase):
    def test_hand_vector_and_termination(self):
        result = encode([1, 0, 1, 1])
        self.assertEqual(result.coded, (1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1))
        self.assertEqual(result.states, (0, 2, 1, 2, 3, 1, 0))
        self.assertEqual(result.inputs_with_tail, (1, 0, 1, 1, 0, 0))
        self.assertEqual(decode_hard(result.coded).information, (1, 0, 1, 1))
        self.assertEqual(decode_soft(bpsk(result.coded)).information, (1, 0, 1, 1))
        self.assertEqual(encode([]).coded, (0, 0, 0, 0))
        self.assertEqual(decode_hard([0, 0, 0, 0]).information, ())

    def test_encoder_matches_independent_polynomial_enumeration(self):
        for length in range(8):
            for information in itertools.product((0, 1), repeat=length):
                result = encode(information)
                self.assertEqual(result.coded, reference_codeword(information))
                self.assertEqual(result.states[0], 0)
                self.assertEqual(result.states[-1], 0)
                self.assertEqual(decode_hard(result.coded).information, information)

    def test_all_hard_observations_short_frames_match_ml_oracle(self):
        # All 64+256+1024 hard vectors for information lengths 1..3.
        for length in range(1, 4):
            words = [
                (bits, reference_codeword(bits))
                for bits in itertools.product((0, 1), repeat=length)
            ]
            for received in itertools.product((0, 1), repeat=2 * (length + 2)):
                minimum = min(
                    sum(a != b for a, b in zip(received, word)) for _, word in words
                )
                result = decode_hard(received)
                self.assertEqual(result.metric, minimum)
                self.assertEqual(
                    sum(
                        a != b
                        for a, b in zip(
                            received, reference_codeword(result.information)
                        )
                    ),
                    minimum,
                )

    def test_soft_short_frames_match_exhaustive_euclidean_ml(self):
        rng = random.Random(8912)
        for length in range(1, 7):
            candidates = [
                (bits, reference_codeword(bits))
                for bits in itertools.product((0, 1), repeat=length)
            ]
            for case in range(20):
                received = [rng.uniform(-2.7, 2.7) for _ in range(2 * (length + 2))]
                costs = [
                    math.fsum(
                        (y - (1 - 2 * bit)) ** 2 for y, bit in zip(received, word)
                    )
                    for _, word in candidates
                ]
                minimum = min(costs)
                result = decode_soft(received)
                selected = math.fsum(
                    (y - (1 - 2 * bit)) ** 2
                    for y, bit in zip(received, reference_codeword(result.information))
                )
                with self.subTest(length=length, case=case):
                    self.assertAlmostEqual(result.metric, minimum, places=10)
                    self.assertAlmostEqual(selected, minimum, places=10)

    def test_traceback_paths_metrics_and_tie_rule(self):
        received = [0.0] * 16
        first = decode_soft(received, trace=True)
        self.assertEqual(first.information, (0,) * 6)
        self.assertEqual(first, decode_soft(received, trace=True))
        self.assertEqual(len(first.metric_history), 9)
        self.assertEqual(len(first.predecessor_history), 8)
        self.assertEqual(first.metric, 16)
        for step, (previous, next_state) in enumerate(
            zip(first.states, first.states[1:])
        ):
            self.assertEqual(
                transition(previous, first.inputs_with_tail[step])[0], next_state
            )
            self.assertEqual(first.predecessor_history[step][next_state], previous)
        self.assertEqual(first.metric_history[-1][0], first.metric)
        self.assertEqual(hard_decision([0.0, -0.1, 0.1]), (0, 1, 0))

    def test_invalid_bits_frames_samples_and_overflow(self):
        for bits in [[2], [True], [0.0], ["1"]]:
            with self.assertRaises(ValueError):
                encode(bits)
        for values in [[], [0, 1], [0] * 5]:
            with self.assertRaises(ValueError):
                decode_hard(values)
            with self.assertRaises(ValueError):
                decode_soft(values)
        for value in [float("nan"), float("inf"), complex(1, 1), 1e308]:
            with self.assertRaises(ValueError):
                decode_soft([value] * 4)
        for state in [-1, 4, True]:
            with self.assertRaises(ValueError):
                transition(state, 0)


class ChannelTests(unittest.TestCase):
    def test_tail_energy_and_information_rate_accounting(self):
        for length in [1, 4, 64, 512]:
            rate = effective_rate(length)
            coded = encode([0] * length)
            self.assertEqual(rate, length / len(coded.coded))
            self.assertEqual(sum(x * x for x in bpsk(coded.coded)) / length, 1 / rate)
            for db in [0, 3, 6]:
                sigma = noise_sigma(db, rate)
                self.assertAlmostEqual(
                    (1 / rate) / (2 * sigma * sigma), 10 ** (db / 10), places=12
                )
        self.assertAlmostEqual(effective_rate(64), 64 / 132)
        self.assertGreater(noise_sigma(3, effective_rate(64)), noise_sigma(3, 1))

    def test_block_wilson_endpoints_and_invalid_counts(self):
        low, high = block_wilson_interval(0, 400)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(
            high, 1.959963984540054**2 / (400 + 1.959963984540054**2), places=14
        )
        a, b = block_wilson_interval(400, 400)
        self.assertAlmostEqual(a, 1 - high)
        self.assertAlmostEqual(b, 1)
        for count, total in [(-1, 10), (11, 10), (0, 0), (1.0, 10)]:
            with self.assertRaises(ValueError):
                block_wilson_interval(count, total)
        for db, rate in [(float("nan"), 0.5), (0, 0), (0, 2), (True, 0.5)]:
            with self.assertRaises(ValueError):
                noise_sigma(db, rate)


if __name__ == "__main__":
    unittest.main()
