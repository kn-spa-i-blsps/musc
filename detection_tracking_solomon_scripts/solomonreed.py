from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List


PRIMITIVE_POLY = 0x11D
GF_EXP = [0] * 512
GF_LOG = [0] * 256

_x = 1
for _i in range(255):
    GF_EXP[_i] = _x
    GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= PRIMITIVE_POLY

for _i in range(255, 512):
    GF_EXP[_i] = GF_EXP[_i - 255]


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("Division by zero in GF(256).")
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] - GF_LOG[b]) % 255]


def poly_eval(coeffs: List[int], x: int) -> int:
    y = 0
    power = 1
    for c in coeffs:
        y ^= gf_mul(c, power)
        power = gf_mul(power, x)
    return y


def solve_vandermonde(xs: List[int], ys: List[int], degree: int) -> List[int]:
    matrix: List[List[int]] = []

    for x, y in zip(xs, ys):
        row: List[int] = []
        power = 1
        for _ in range(degree):
            row.append(power)
            power = gf_mul(power, x)
        row.append(y)
        matrix.append(row)

    n = degree

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if matrix[row][col] != 0:
                pivot = row
                break

        if pivot is None:
            raise ValueError("Singular matrix.")

        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]

        inv_pivot = gf_div(1, matrix[col][col])
        for c in range(col, n + 1):
            matrix[col][c] = gf_mul(matrix[col][c], inv_pivot)

        for row in range(n):
            if row == col or matrix[row][col] == 0:
                continue
            factor = matrix[row][col]
            for c in range(col, n + 1):
                matrix[row][c] ^= gf_mul(factor, matrix[col][c])

    return [matrix[i][n] for i in range(n)]


class ReedSolomonDecodeError(Exception):
    pass


class SalomonReed:
    def __init__(self, redundancy: float = 2.0) -> None:
        self.set(redundancy)

    def set(self, redundancy: float) -> None:
        redundancy = float(redundancy)
        if redundancy <= 0:
            raise ValueError("Redundancy must be > 0.")
        self.redundancy = redundancy

    def parameters(self, input_bits: int) -> Dict[str, float]:
        if input_bits <= 0 or input_bits % 8 != 0:
            raise ValueError("Input length must be a positive multiple of 8 bits.")

        k = input_bits // 8
        p = math.ceil(self.redundancy * k)
        n = k + p

        if n > 255:
            raise ValueError("Too many symbols for GF(256).")

        return {
            "input_bits": input_bits,
            "data_symbols": k,
            "parity_symbols": p,
            "total_symbols": n,
            "encoded_bits": n * 8,
            "code_rate": k / n,
            "correctable_symbol_errors": p // 2,
        }

    def encode(self, bits: str) -> str:
        data = self._bits_to_bytes(bits)
        _, _, n = self._sizes_from_k(len(data))
        xs = list(range(1, n + 1))
        code_symbols = [poly_eval(data, x) for x in xs]
        return self._bytes_to_bits(code_symbols)

    def decode(self, bits: str) -> str:
        received = self._bits_to_bytes(bits)
        n = len(received)

        k, p = self._infer_k_and_p_from_n(n)
        t = p // 2

        xs = list(range(1, n + 1))
        best_coeffs = None
        best_mismatches = n + 1

        for subset in combinations(range(n), k):
            sel_x = [xs[i] for i in subset]
            sel_y = [received[i] for i in subset]

            try:
                coeffs = solve_vandermonde(sel_x, sel_y, k)
            except ValueError:
                continue

            predicted = [poly_eval(coeffs, x) for x in xs]
            mismatches = sum(
                1 for expected, got in zip(predicted, received) if expected != got
            )

            if mismatches < best_mismatches:
                best_mismatches = mismatches
                best_coeffs = coeffs
                if mismatches == 0:
                    break

        if best_coeffs is None or best_mismatches > t:
            raise ReedSolomonDecodeError(
                f"Too many errors: found at least {best_mismatches}, maximum correctable is {t}."
            )

        return self._bytes_to_bits(best_coeffs)

    def _sizes_from_k(self, k: int) -> tuple[int, int, int]:
        p = math.ceil(self.redundancy * k)
        n = k + p

        if n > 255:
            raise ValueError("Too many symbols for GF(256).")

        return k, p, n

    def _infer_k_and_p_from_n(self, n: int) -> tuple[int, int]:
        candidates = []
        for k in range(1, n + 1):
            p = math.ceil(self.redundancy * k)
            if k + p == n:
                candidates.append((k, p))

        if len(candidates) != 1:
            raise ValueError(
                f"Cannot infer original message length uniquely from n={n} and redundancy={self.redundancy}."
            )

        return candidates[0]

    @staticmethod
    def _bits_to_bytes(bits: str) -> List[int]:
        if not bits:
            raise ValueError("Empty input.")
        if any(ch not in "01" for ch in bits):
            raise ValueError("Input must contain only '0' and '1'.")
        if len(bits) % 8 != 0:
            raise ValueError("Input length must be a multiple of 8 bits.")

        return [int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)]

    @staticmethod
    def _bytes_to_bits(data: List[int]) -> str:
        return "".join(f"{byte:08b}" for byte in data)