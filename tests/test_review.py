"""Fixed channel observations independently derived from the hand codeword."""

import unittest
from convcode.core import decode_hard, decode_soft, hard_decision


class ReviewTests(unittest.TestCase):
    def test_soft_reliability_recovers_hand_codeword_after_three_weak_sign_errors(self):
        # Payload1011 + zero tails: g7,g5 pairs 11 10 00 01 01 11.
        coded = (1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1)
        received = [float(1 - 2 * bit) for bit in coded]
        for index in (0, 1, 2):
            received[index] *= -0.05
        hard = decode_hard(hard_decision(received))
        soft = decode_soft(received, trace=True)
        self.assertNotEqual(hard.information, (1, 0, 1, 1))
        self.assertEqual(soft.information, (1, 0, 1, 1))
        self.assertAlmostEqual(soft.metric, 3 * 1.05**2, places=13)
        self.assertEqual(soft.states, (0, 2, 1, 2, 3, 1, 0))
        self.assertEqual(soft.inputs_with_tail[-2:], (0, 0))

    def test_known_final_state_overrides_cheaper_unterminated_path(self):
        # Input11 from state00 emits11,01 and ends11. A four-sample terminated
        # frame has no information bits, so the only legal input path is00,
        # emitting00,00. Selecting the best arbitrary final state would be wrong.
        hard = decode_hard((1, 1, 0, 1), trace=True)
        soft = decode_soft((-1, -1, 1, -1), trace=True)
        self.assertEqual(hard.metric_history[-1][3], 0)
        self.assertEqual(soft.metric_history[-1][3], 0)
        for result in (hard, soft):
            self.assertEqual(result.information, ())
            self.assertEqual(result.states, (0, 0, 0))
            self.assertEqual(result.inputs_with_tail, (0, 0))
        self.assertEqual(hard.metric, 3)
        self.assertEqual(soft.metric, 12)
