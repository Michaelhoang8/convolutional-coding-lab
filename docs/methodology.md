# Simulation methodology

## Channel and information energy

All signals are created here. Independent equiprobable payload bits are mapped to real BPSK symbols by `0 → +1`, `1 → -1`, so symbol energy `Es = 1`. Independent zero-mean Gaussian noise is added directly to the ideal sampled symbol. There is no sampled carrier, clock recovery, pulse shaping, fading or hardware capture.

For payload length `L`, the terminated code transmits `2(L+2)` symbols. Its effective information rate is `R_eff = L/[2(L+2)]`; uncoded rate is 1. Total frame energy divided by information bits gives `Eb,info = Es/R_eff`. At target `gamma = 10^(Eb(info)/N0[dB]/10)`, real noise variance is:

```text
sigma² = N0/2 = 1 / (2 * R_eff * gamma)
```

This charges both zero-tail inputs and all four resulting tail symbols. For `L=64`, effective rate is `64/132 ≈ 0.484848` and energy per information bit is `2.0625`. At 3 dB the uncoded noise standard deviation is approximately `0.500593`, versus `0.718922` for coded transmission. Comparing both with the same variance would silently give the coded run more energy per information bit.

This comparison holds information-bit energy relative to noise constant, not bandwidth or throughput. The analytic uncoded reference is `Pb = Q(sqrt(2 gamma)) = 0.5 * erfc(sqrt(gamma))`, for ideal coherent BPSK in AWGN. See the [University of Michigan EECS 555 modulation notes, chapter 6, page 6-3](https://www.eecs.umich.edu/courses/eecs555/chap6.pdf), including the real noise variance `N0/2`.

## Frame design and seeds

There are 400 frames at each of 0, 1.5, 3, 4.5 and 6 dB, each containing 64 information bits. The design is fixed in advance, with no stop-on-error or best-seed selection in the aggregate run.

```text
bit_seed = 20261121 + snr_index * 100000 + frame_index
uncoded_noise_seed = bit_seed + 1000000
coded_noise_seed = bit_seed + 2000000
```

Each frame initializes separate `random.Random` instances. Bits use `getrandbits(1)`; noise uses `gauss(0, sigma)`. These are deterministic pseudorandom draws, treated as independent under the ideal simulation model. Python and platform versions are recorded. Math-library differences may change Gaussian draws by tiny rounding amounts across machines; the actual measured observations are preserved separately, without regeneration.

Uncoded and coded methods use the same payload and different noise streams, because their symbol counts and variance differ. Hard and soft decoders share the identical coded noisy samples. Their outcomes are paired, not independent method samples.

`demo.json` is a separate eight-bit illustration at 2 dB. It is the first seed in `0..4095` for which hard decoding has errors and soft decoding has none (the published example uses seed 0). This explicit selection makes it useful for explaining confidence weighting but unsuitable for estimating an average advantage. It is excluded from all aggregate BER, block errors and timing statistics.

## Metrics and uncertainty

Information-bit BER divides post-decoding information errors by `400 × 64 = 25,600` at each method/point. It excludes the two tail inputs. Pre-decoding sliced coded-channel errors are saved separately, divided by `400 × 132 = 52,800` if a channel-symbol rate is calculated. Do not compare these denominators interchangeably.

A frame-error indicator is one when at least one of its 64 information bits is wrong. At a fixed signal level, the independent equal-length frames support a Bernoulli frame-error model. Wilson 95% intervals use `z = 1.959963984540054`, with zero/all-error numerical endpoints set exactly to 0/1. These intervals have approximate frequentist coverage, not a guaranteed probability statement for each realized interval. The original score-interval construction is [Wilson (1927), JASA 22(158), 209–212](https://doi.org/10.1080/01621459.1927.10502953), also available as an [original-paper scan](https://www.jhanley.biostat.mcgill.ca/c607/ch08/wilson_jasa_1927.pdf).

Decoded bit errors within a frame are correlated. No binomial bitwise intervals are placed on the coded information BER. Method-specific frame intervals also do not constitute an interval for the paired hard-minus-soft difference. No multiple-comparison inference or significance claim is made. A zero BER count is omitted from the logarithmic axis and labeled explicitly; it is not replaced by an invented probability floor.

## Timing

Each method gets one warmup call. An independent RNG with seed 9182 randomizes method order within each frame. `perf_counter_ns` measures a complete method call: uncoded slicing, coded slicing plus hard Viterbi, or soft Viterbi. Noise generation, encoding, record assembly and disk I/O are outside the timer. Each observation is timed once, for 6,000 actual calls on 2,000 distinct frames.

The summary reports medians and full observed ranges over 400 calls at each method/point. This short interpreter measurement has scheduler and timer noise, and no controlled CPU frequency or affinity. It is not a modem throughput benchmark or real-time guarantee. New runs are expected to have different timings.

## Evidence formats and hashes

`received.f64` contains headerless little-endian IEEE754 float64 values. For each row of `frames.csv`, read `sample_byte_length` bytes starting at `sample_byte_offset`. The first `information_length` doubles are uncoded observations; the rest are coded observations shared by hard and soft. Per-frame SHA256 verifies that exact byte block. The published binary is 3,136,000 bytes.

The CSV preserves bit strings and decimal state strings (`0..3`), not integers, so leading zeroes matter. Both decoded information errors and block indicators are stored for reconciliation. CSV/JSON evidence is written with LF line endings before exact-byte hashing. Source hashes normalize CRLF/CR to LF; the floating-point binary and PNG/MP4 media are never text-normalized. `.gitattributes` marks these formats binary explicitly.

`manifest.json` records exact byte sizes and SHA256 for every run artifact except itself. `environment.json` captures source hashes **before** simulation. Figure metadata links exact input hashes to image/report hashes; replay metadata additionally links the saved demo and normalized renderer source to the actual video/poster hashes. The test suite verifies the links and re-decodes all preserved samples.
