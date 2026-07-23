"""Spin-1/2 finite-period TDVP equations and quantum leakage."""

from __future__ import annotations

import numpy as np


def _eta_from_narr(narr: np.ndarray) -> np.ndarray:
    """Evaluate all cyclic environment weights in O(K) work.

    ``narr`` may have shape ``(K,)`` or ``(K, n_times)``.  The weights obey

    ``eta[i] = 1 + narr[i-1] * eta[i-1]``

    with the periodic value of ``eta[0]`` fixed by one product around the cell.
    """

    narr = np.asarray(narr)
    period = narr.shape[0]
    beta = np.prod(narr, axis=0)
    backward_indices = (-np.arange(1, period + 1)) % period
    backward_products = np.cumprod(narr[backward_indices], axis=0)
    eta = np.empty_like(narr, dtype=np.result_type(narr, np.float64))
    eta[0] = 1.0 + np.sum(backward_products, axis=0) / (1.0 - beta)
    for site in range(1, period):
        eta[site] = 1.0 + narr[site - 1] * eta[site - 1]
    return eta


def get_eta(theta: np.ndarray) -> np.ndarray:
    """Return the spin-1/2 transfer weights for ``theta``."""

    return _eta_from_narr(-np.sin(np.asarray(theta) / 2.0) ** 2)


def get_qleak(thetaphi: np.ndarray) -> np.ndarray:
    """Return the intensive leakage amplitude Gamma.

    The first axis contains ``theta`` followed by ``phi``.  The spin-1/2
    expression is independent of ``phi`` but the shared layout keeps calls
    consistent with the ODE trajectory.
    """

    thetaphi = np.asarray(thetaphi)
    period = thetaphi.shape[0] // 2
    theta = thetaphi[:period]
    eta = get_eta(theta)
    eta_next = np.roll(eta, -1, axis=0)
    theta_next = np.roll(theta, -1, axis=0)
    gamma_squared = np.mean(
        np.sin(theta / 2.0) ** 2
        * np.sin(theta_next / 2.0) ** 2
        * eta
        * (1.0 - eta)
        / eta_next,
        axis=0,
    )
    return np.sqrt(np.abs(gamma_squared))


def eom(
    _time: float,
    state: np.ndarray,
    mu: float,
    chi: float,
) -> np.ndarray:
    """Evaluate the spin-1/2 TDVP flow for staggered detuning."""

    period = state.size // 2
    if state.size != 2 * period or period % 2 != 0:
        raise ValueError("the staggered-detuning unit cell must have even K")
    theta = state[:period]
    phi = state[period:]

    eta = get_eta(theta)
    eta_previous = np.roll(eta, 1)
    eta_next = np.roll(eta, -1)
    theta_next = np.roll(theta, -1)
    theta_next2 = np.roll(theta, -2)
    theta_previous = np.roll(theta, 1)
    phi_next = np.roll(phi, -1)
    phi_previous = np.roll(phi, 1)
    detuning = 2.0 * mu + chi * np.where(
        np.arange(period) % 2 == 0,
        -1.0,
        1.0,
    )

    theta_dot = (
        2.0 * np.cos(theta_next / 2.0) * np.sin(phi)
        + eta_previous
        * np.sin(theta_previous)
        * np.sin(theta / 2.0)
        * np.sin(phi_previous)
        / eta
    )

    phi_dot = (
        2.0 * np.cos(theta_next / 2.0) * np.cos(phi) / np.tan(theta)
        + 2.0 * detuning
        - np.cos(theta_next2 / 2.0)
        * np.cos(phi_next)
        * np.tan(theta_next / 2.0)
        - eta_previous
        * np.sin(theta_previous)
        * np.cos(phi_previous)
        / (2.0 * eta * np.cos(theta / 2.0))
        - eta
        * np.sin(theta)
        * np.cos(phi)
        * np.sin(theta_next / 2.0)
        * np.tan(theta_next / 2.0)
        / (2.0 * eta_next)
    )
    return np.concatenate((theta_dot, phi_dot))
