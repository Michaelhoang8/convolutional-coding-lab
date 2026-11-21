# Findings

These observations refer to `results/published`, a synthetic 2,000-frame run with 64 information bits per frame. Rates include the energy of all termination symbols. Results are measured point estimates, not universal coding-gain claims.

## Information-bit errors

| Eb(info)/N0 | Uncoded errors | Hard Viterbi errors | Soft Viterbi errors |
|---:|---:|---:|---:|
| 0 dB | 1,999 | 5,160 | 2,514 |
| 1.5 dB | 1,252 | 2,464 | 706 |
| 3 dB | 581 | 740 | 68 |
| 4.5 dB | 254 | 150 | 1 |
| 6 dB | 68 | 29 | 0 |

Every cell has 25,600 information bits in its denominator. At 3 dB the observed soft BER was `0.00265625`, compared with `0.02890625` for hard decoding and `0.0226953125` uncoded. The result supports retaining observation confidence for this code and channel. At 0 dB both coded methods had worse information BER than uncoded transmission; coding overhead and low-confidence path decisions matter.

The uncoded results are broadly consistent with the analytic AWGN reference, without being identical finite-sample realizations. At 3 dB, measured uncoded BER is `0.0226953` versus theoretical `0.0228784`. This is a calibration check, not a proof that all simulation details are correct.

## Bit errors and frame errors differ

At 3 dB, hard decoding produced more erroneous information bits than uncoded detection, but fewer failed frames: 201 versus 303. A failed Viterbi path can contain several bit errors in one frame. Soft decoding failed 29 frames at that point. This is why coded-bit errors should not be treated as 25,600 independent Bernoulli trials for a confidence interval.

At 6 dB, hard decoding failed 12 of 400 frames and soft decoding failed none. The soft Wilson interval for frame-error probability is `[0, 0.00951229]`, under the stated independent-frame model. This is not a BER bound, and the small run cannot establish rare-event performance. More frames and a declared stopping/inference design would be needed for that claim.

## Runtime and implementation tradeoffs

At 3 dB, observed median complete-call times were 7.6 µs for uncoded slicing, 122.4 µs for hard decoding including slicing, and 163.1 µs for soft decoding. The corresponding observed ranges were 7.2–14.4, 117.5–211.5 and 156.4–292.6 µs. The actual platform was Windows/AMD64, Python 3.13.17; every raw duration is retained.

Soft decoding performs floating-point branch-distance calculations, while hard decoding counts mismatches. Both use four states and full-frame traceback; the source prioritizes explicit metric and termination behavior over throughput. Per-call interpreter and scheduler effects limit cross-system conclusions. The replay deliberately takes eight seconds to explain a computation measured in microseconds; it is not a live decoder or timing visualization.

## Limits

The study fixes one short code and one payload length. It does not measure fading, burst channels, phase errors, timing offset, finite-precision soft metrics, puncturing, interleaving, alternate codes or practical modem bandwidth. It does not implement posterior bitwise decoding or hardware acceleration. The illustrative trellis frame is selected for a hard/soft contrast and excluded from estimates. All data and code were created with AI assistance; there are no claims of physical radio operation, client impact or external dataset provenance.
