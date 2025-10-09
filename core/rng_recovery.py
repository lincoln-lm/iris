from typing import Iterable
import numpy as np
from numba_pokemon_prngs.xorshift import Xorshift128
from .matrix_util import (
    co_kernel_basis,
    generalized_inverse,
    int_to_bit_vector,
    bit_vector_to_int,
)


def reverse_float(minimum: float, maximum: float, value: float):
    """Go from a randrange float to the truncated bits that produced it"""

    return int(((value - maximum) / (minimum - maximum)) * 0x7FFFFF) & 0x7FFFFF


def extract_known_bit_length_float(
    minimum: float, maximum: float, leeway: float, rand: float
) -> int:
    """Extract the number of known bits from a randrange float given a particular leeway from the true value"""
    min_interval = max(rand - leeway, minimum)
    max_interval = min(rand + leeway, maximum)
    min_raw = reverse_float(minimum, maximum, min_interval)
    max_raw = reverse_float(minimum, maximum, max_interval)
    # the string of bits that are the same (starting from the MSB)
    # in the min and max of the range are guaranteed to be the same
    # throughout the whole range,so must be shared by the true value
    same = ~(min_raw ^ max_raw) & 0x7FFFFF
    known_bit_length = 0
    for known_bit_length in range(23):
        if not (same >> (22 - known_bit_length)) & 1:
            break

    return known_bit_length


def extract_known_bits_float(
    minimum: float, maximum: float, leeway: float, rands: Iterable[float]
) -> tuple[np.ndarray, list[int]]:
    """
    Extract the bits from each randrange float that can be confidently determined
    given a particular leeway from the true value
    """

    known_bits = np.zeros(0, dtype=np.uint8)
    known_bit_lengths = []
    for rand in rands:
        rand_raw = reverse_float(minimum, maximum, rand)
        known_bit_length = extract_known_bit_length_float(
            minimum, maximum, leeway, rand
        )
        new_known_bits = rand_raw >> (23 - known_bit_length)
        # concating every time is wasteful but the alternative is inelegant
        known_bits = np.concatenate(
            (known_bits, int_to_bit_vector(new_known_bits, known_bit_length))
        )
        known_bit_lengths.append(known_bit_length)
    return known_bits, known_bit_lengths


def build_observation_mat_float(bit_lengths: list[int]) -> np.ndarray:
    """
    Build a matrix mapping the rng state to the higher bits of consecutive float randranges
    """

    total_output = sum(bit_lengths)
    observation_matrix = np.zeros((128, total_output), dtype=np.uint8)

    # [1, 0, ...] @ mat = mat[0] = the observations when the state is 0b1
    # [0, 1, 0, ...] @ mat = mat[1] = the observations when the state is 0b10
    # etc.
    # using this fact builds the observation matrix much simpler than doing it explicitly
    for state_bit in range(128):
        state_int = 1 << state_bit
        state = (
            state_int & 0xFFFFFFFF,
            (state_int >> 32) & 0xFFFFFFFF,
            (state_int >> 64) & 0xFFFFFFFF,
            (state_int >> 96) & 0xFFFFFFFF,
        )
        rng = Xorshift128(*state)
        observation_bit = 0
        for bit_length in bit_lengths:
            raw = int(rng.next() & 0x7FFFFF)
            observed = raw >> (23 - bit_length)
            observation_matrix[
                state_bit, observation_bit : observation_bit + bit_length
            ] = int_to_bit_vector(observed, bit_length)
            observation_bit += bit_length

    return observation_matrix


def recover_rng_state_float(
    minimum: float, maximum: float, leeway: float, rands: Iterable[float]
) -> tuple[int, int, int, int] | None:
    known_bits, known_bit_lengths = extract_known_bits_float(
        minimum, maximum, leeway, rands
    )
    observation_matrix = build_observation_mat_float(known_bit_lengths)
    # the system is not completely determined, more info needed
    if co_kernel_basis(observation_matrix).any():
        return None
    # map from observations -> state
    inverse = generalized_inverse(observation_matrix)
    state_vec = (known_bits @ inverse) & 1
    state_int = bit_vector_to_int(state_vec)
    return (
        state_int & 0xFFFFFFFF,
        (state_int >> 32) & 0xFFFFFFFF,
        (state_int >> 64) & 0xFFFFFFFF,
        (state_int >> 96) & 0xFFFFFFFF,
    )
