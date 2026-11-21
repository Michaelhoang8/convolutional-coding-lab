"""Owned seeded AWGN frame generator with explicit information-bit energy."""

import random

from .core import bpsk, encode, effective_rate, noise_sigma

BASE_SEED = 20261121
EBN0_DB = (0.0, 1.5, 3.0, 4.5, 6.0)


def frame(snr_index, frame_index, information_length=64):
    if type(snr_index) is not int or not 0 <= snr_index < len(EBN0_DB):
        raise ValueError("Invalid SNR index")
    if type(frame_index) is not int or not 0 <= frame_index < 100000:
        raise ValueError("Frame index must lie in 0..99999")
    rate = effective_rate(information_length)
    seed = BASE_SEED + snr_index * 100000 + frame_index
    bit_rng = random.Random(seed)
    information = tuple(bit_rng.getrandbits(1) for _ in range(information_length))
    encoded = encode(information)
    uncoded_seed = seed + 1000000
    coded_seed = seed + 2000000
    uncoded_rng = random.Random(uncoded_seed)
    coded_rng = random.Random(coded_seed)
    uncoded_sigma = noise_sigma(EBN0_DB[snr_index], 1)
    coded_sigma = noise_sigma(EBN0_DB[snr_index], rate)
    uncoded = tuple(
        symbol + uncoded_rng.gauss(0, uncoded_sigma) for symbol in bpsk(information)
    )
    coded = tuple(
        symbol + coded_rng.gauss(0, coded_sigma) for symbol in bpsk(encoded.coded)
    )
    return {
        "information": information,
        "encoded": encoded,
        "uncoded_received": uncoded,
        "coded_received": coded,
        "bit_seed": seed,
        "uncoded_noise_seed": uncoded_seed,
        "coded_noise_seed": coded_seed,
        "uncoded_sigma": uncoded_sigma,
        "coded_sigma": coded_sigma,
        "effective_rate": rate,
        "ebn0_db": EBN0_DB[snr_index],
    }
