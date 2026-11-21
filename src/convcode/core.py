"""Source-built rate-1/2 K=3 encoder and full-frame hard/soft Viterbi decoder."""

from dataclasses import dataclass
import math


def _bits(values):
    result = tuple(values)
    if any(type(value) is not int or value not in (0, 1) for value in result):
        raise ValueError("Bits must be integers 0 or 1")
    return result


def _samples(values):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Samples must be finite real values") from exc
    if any(not math.isfinite(value) for value in result):
        raise ValueError("Samples must be finite real values")
    return result


def transition(state, input_bit):
    """State = (previous input << 1) | older input; output order is (7,5)."""
    if (
        type(state) is not int
        or not 0 <= state < 4
        or type(input_bit) is not int
        or input_bit not in (0, 1)
    ):
        raise ValueError("State must lie in 0..3 and input must be a bit")
    previous, older = (state >> 1) & 1, state & 1
    output = (input_bit ^ previous ^ older, input_bit ^ older)
    return (input_bit << 1) | previous, output


TRANSITIONS = tuple(
    tuple(transition(state, bit) for bit in (0, 1)) for state in range(4)
)


@dataclass(frozen=True)
class Encoded:
    information: tuple[int, ...]
    inputs_with_tail: tuple[int, ...]
    coded: tuple[int, ...]
    states: tuple[int, ...]


def encode(information):
    """Start at state 00, append two zero inputs, and return to state 00."""
    information = _bits(information)
    inputs = information + (0, 0)
    state = 0
    states = [0]
    coded = []
    for bit in inputs:
        state, output = TRANSITIONS[state][bit]
        coded.extend(output)
        states.append(state)
    assert state == 0
    return Encoded(information, inputs, tuple(coded), tuple(states))


def bpsk(bits):
    """Unit symbol-energy mapping: bit 0 -> +1 and bit 1 -> -1."""
    return tuple(1 - 2 * bit for bit in _bits(bits))


def hard_decision(samples):
    """Zero threshold; an exact zero sample is assigned bit 0."""
    return tuple(int(value < 0) for value in _samples(samples))


@dataclass(frozen=True)
class Decoded:
    information: tuple[int, ...]
    inputs_with_tail: tuple[int, ...]
    states: tuple[int, ...]
    metric: float
    metric_history: tuple[tuple[float | None, ...], ...]
    predecessor_history: tuple[tuple[int | None, ...], ...]


def _viterbi(received, soft, trace):
    observations = _samples(received) if soft else _bits(received)
    if len(observations) < 4 or len(observations) % 2:
        raise ValueError(
            "A terminated frame requires an even length of at least four coded samples"
        )
    metrics = [0.0, math.inf, math.inf, math.inf]
    predecessors = []
    histories = [(0.0, None, None, None)] if trace else []
    for offset in range(0, len(observations), 2):
        pair = observations[offset : offset + 2]
        updated = [math.inf] * 4
        incoming = [None] * 4
        # Strict improvement keeps the first tie: lower predecessor state,
        # then lower input bit, from this explicit iteration order.
        for state in range(4):
            if math.isinf(metrics[state]):
                continue
            for bit in (0, 1):
                next_state, output = TRANSITIONS[state][bit]
                if soft:
                    try:
                        branch = (pair[0] - (1 - 2 * output[0])) ** 2 + (
                            pair[1] - (1 - 2 * output[1])
                        ) ** 2
                    except OverflowError as exc:
                        raise ValueError("Soft metric overflow") from exc
                else:
                    branch = int(pair[0] != output[0]) + int(pair[1] != output[1])
                candidate = metrics[state] + branch
                if not math.isfinite(candidate):
                    raise ValueError("Path metric overflow")
                if candidate < updated[next_state]:
                    updated[next_state] = candidate
                    incoming[next_state] = state
        predecessors.append(tuple(incoming))
        metrics = updated
        if trace:
            histories.append(
                tuple(value if math.isfinite(value) else None for value in metrics)
            )
    # Termination is known: choose state 00, not the best arbitrary final state.
    state = 0
    states = [0]
    inputs = []
    for incoming in reversed(predecessors):
        inputs.append(state >> 1)
        state = incoming[state]
        if state is None:
            raise ValueError("No legal terminated path")
        states.append(state)
    states.reverse()
    inputs.reverse()
    assert states[0] == 0 and states[-1] == 0 and inputs[-2:] == [0, 0]
    return Decoded(
        tuple(inputs[:-2]),
        tuple(inputs),
        tuple(states),
        metrics[0],
        tuple(histories),
        tuple(predecessors) if trace else (),
    )


def decode_hard(received, trace=False):
    """ML terminated path for a BSC with crossover probability below 1/2."""
    return _viterbi(received, False, trace)


def decode_soft(received, trace=False):
    """Maximum-likelihood terminated path for equal-variance AWGN BPSK samples."""
    return _viterbi(received, True, trace)


def effective_rate(information_bits):
    if type(information_bits) is not int or information_bits <= 0:
        raise ValueError("Information length must be a positive integer")
    return information_bits / (2 * (information_bits + 2))


def noise_sigma(ebn0_db, rate):
    """For Es=1, Eb(information)/N0=gamma: sigma² = 1/(2*rate*gamma)."""
    if (
        isinstance(ebn0_db, bool)
        or not isinstance(ebn0_db, (int, float))
        or not math.isfinite(ebn0_db)
    ):
        raise ValueError("Eb/N0 must be finite")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or not 0 < rate <= 1
    ):
        raise ValueError("Effective rate must lie in (0,1]")
    try:
        variance = 1 / (2 * rate * 10 ** (ebn0_db / 10))
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Noise scale outside numeric range") from exc
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("Noise scale outside numeric range")
    return math.sqrt(variance)


def block_wilson_interval(errors, frames, z=1.959963984540054):
    """95% Wilson interval for independent equal-distribution frame-error events.

    This function is used only for block outcomes, not correlated decoded bits.
    """
    if (
        type(errors) is not int
        or type(frames) is not int
        or not 0 <= errors <= frames
        or frames <= 0
    ):
        raise ValueError("Require integer 0 <= errors <= positive frame count")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    p = errors / frames
    denominator = 1 + z * z / frames
    center = (p + z * z / (2 * frames)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / frames + z * z / (4 * frames * frames))
        / denominator
    )
    low = 0.0 if errors == 0 else max(0.0, center - radius)
    high = 1.0 if errors == frames else min(1.0, center + radius)
    return low, high
