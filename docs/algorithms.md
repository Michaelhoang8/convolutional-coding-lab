# Algorithms

## Encoder convention

For current bit `u`, previous bit `p`, and older bit `q`, the state integer is `2p + q`. The two output bits are:

```text
g7 = u XOR p XOR q
g5 = u XOR q
next_state = 2u + p
```

The generator polynomials are `1 + D + D²` and `1 + D²`, written as octal `(7,5)`. Output order matters: generator 7 first, generator 5 second. Start in state `00`; append two zero inputs so the final state is also `00`. A payload of length `L` produces `2(L+2)` coded bits.

| State | Input 0: output → next | Input 1: output → next |
|---|---|---|
| 00 | 00 → 00 | 11 → 10 |
| 01 | 11 → 00 | 00 → 10 |
| 10 | 10 → 01 | 01 → 11 |
| 11 | 01 → 01 | 10 → 11 |

The hand vector `1011` followed by `00` gives pairs `11 10 00 01 01 11` and states `00,10,01,10,11,01,00`. Independent tests generate codewords using polynomial dot products over a shift register rather than the production transition table.

## Viterbi recurrence

Initialize state-00 metric to zero and other state metrics to infinity. For each received pair, consider both outgoing branches from each reachable predecessor. Retain the minimum cumulative metric for each destination state and record its predecessor. Trace backward from **state 00**, even if another final state has a smaller metric. The final two recovered inputs are tails; remove them from the returned information bits.

The principle is the finite-state shortest-path recurrence used by Viterbi decoding. The implementation follows the recurrence directly and does not call an external decoder. See [Viterbi's original paper, IEEE Transactions on Information Theory 13(2), 260–269 (1967)](https://doi.org/10.1109/TIT.1967.1054010), and [MIT 6.02 lecture 7](https://web.mit.edu/6.02/www/currentsemester/handouts/L07_notes_pictures.pdf) for the trellis and survivor interpretation.

Hard decoding uses the number of unequal bits in each pair. For a memoryless binary symmetric channel with crossover probability `0 < p < 1/2`, minimizing Hamming distance maximizes codeword likelihood. At `p=1/2` all codewords have equal likelihood; at `p>1/2` the same minimization is not the ML rule. The simulated finite-variance BPSK hard slicer has crossover probability below one half.

Soft decoding uses `sum((y - (1 - 2c))²)` for each coded pair. Independent, equal-variance Gaussian samples have a log-likelihood differing from this distance only by a positive scale and a constant. Thus the Viterbi path minimizes the full terminated-codeword Euclidean distance. This is sequence ML; it does not minimize each bit's posterior error probability independently. A posterior bitwise decoder such as BCJR is outside scope.

## Numerical and boundary behavior

- Exact metric ties retain the first candidate visited: lower predecessor state, then lower input bit. Equal final metrics are still resolved by the required final state, not by arbitrary state selection.
- The hard slicer maps negative values to 1 and exact zero to 0. Soft input must be finite; nonfinite values and overflowing cumulative metrics are rejected.
- Encoded observations require an even length of at least four. Four samples encode a legal empty payload with only two tail inputs.
- Traces preserve every forward state metric and every survivor pointer. Unreachable metrics appear as JSON `null`; stored state paths include initial and final states.
- Runtime is `O(4 × 2 × (L+2))`, linear in frame length for this fixed code. Full traceback memory is `O(4 × (L+2))`. This is a block decoder, not bounded-memory streaming traceback. The reusable Python API does not impose a resource quota; study dimensions are bounded explicitly.

## Validation independence

For all payload lengths 0 through 7, the production encoder matches an independent polynomial encoder. For lengths 1 through 3, all 1,344 possible hard received sequences are compared with brute-force distances to every legal codeword. Another 120 seeded soft observations at lengths 1 through 6 are compared with independently summed Euclidean distances. Ties are checked for equal optimal metric, not forced to an arbitrary oracle payload.

The final-state test constructs a received sequence perfectly matching an unterminated path. The legal final-state-00 decoder must instead pay Hamming metric 3 or Euclidean metric 12. The confidence test flips the signs of three low-magnitude samples: hard decoding fails while soft decoding recovers the independently specified hand codeword.
