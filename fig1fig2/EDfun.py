"""Minimal exact-diagonalization helpers used by the figure reproducer."""

from __future__ import annotations

import math

import numpy as np


def mpsmanifold(theta: np.ndarray, phi: np.ndarray, basis) -> np.ndarray:
    """Expand the blockade-compatible bond-dimension-2 MPS in ``basis``.

    The function returns one normalized row vector.  ``theta`` and ``phi``
    specify one variational unit cell; the cell must divide the ring length.
    """

    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if theta.shape != phi.shape or theta.ndim != 1:
        raise ValueError("theta and phi must be one-dimensional arrays")

    number_of_states = basis.Ns
    local_dimension = round(basis.sps)
    ring_length = basis.N
    period = theta.size
    if ring_length % period != 0:
        raise ValueError("the variational period must divide the ring length")

    occupations = np.asarray(
        [
            [
                (basis[state_index] // local_dimension ** (ring_length - site - 1))
                % local_dimension
                for site in range(ring_length)
            ]
            for state_index in range(number_of_states)
        ],
        dtype=int,
    )

    tensors = np.zeros(
        (ring_length, local_dimension, 2, 2),
        dtype=complex,
    )
    for site in range(ring_length):
        local_theta = theta[site % period]
        local_phi = phi[site % period]
        tensors[site, 0] = np.asarray(
            [[np.cos(local_theta / 2.0) ** (local_dimension - 1), 0.0],
             [1.0, 0.0]]
        )
        for occupation in range(1, local_dimension):
            binomial = math.factorial(local_dimension - 1) / (
                math.factorial(local_dimension - 1 - occupation)
                * math.factorial(occupation)
            )
            amplitude = (
                math.sqrt(binomial)
                * np.cos(local_theta / 2.0)
                ** (local_dimension - 1 - occupation)
                * (
                    np.sin(local_theta / 2.0)
                    * np.exp(-1j * local_phi)
                )
                ** occupation
            )
            tensors[site, occupation] = np.asarray(
                [[0.0, amplitude], [0.0, 0.0]]
            )

    coefficients = np.zeros((1, number_of_states), dtype=complex)
    for state_index, occupation_row in enumerate(occupations):
        product = np.eye(2, dtype=complex)
        for site, occupation in enumerate(occupation_row):
            product = product @ tensors[site, occupation]
        coefficients[0, state_index] = np.trace(product)

    norm = np.linalg.norm(coefficients)
    if norm == 0.0:
        raise ValueError("the variational state has zero norm")
    return coefficients / norm
