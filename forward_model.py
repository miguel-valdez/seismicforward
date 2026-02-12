"""Core 1D forward-modeling utilities for seismic layered media."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LayerModel:
    """A simple horizontally layered acoustic model.

    Attributes
    ----------
    thicknesses:
        Layer thicknesses in meters for N layers.
    velocities:
        P-wave velocity for each layer in m/s, length N.
    densities:
        Density for each layer in kg/m^3, length N.
    """

    thicknesses: np.ndarray
    velocities: np.ndarray
    densities: np.ndarray

    @classmethod
    def from_iterables(
        cls,
        thicknesses: Iterable[float],
        velocities: Iterable[float],
        densities: Iterable[float],
    ) -> "LayerModel":
        t = np.asarray(list(thicknesses), dtype=float)
        v = np.asarray(list(velocities), dtype=float)
        rho = np.asarray(list(densities), dtype=float)

        if t.ndim != 1 or v.ndim != 1 or rho.ndim != 1:
            raise ValueError("Model arrays must be one-dimensional.")
        if len(t) == 0:
            raise ValueError("At least one layer is required.")
        if len(t) != len(v) or len(v) != len(rho):
            raise ValueError("thicknesses, velocities, and densities must have equal length.")
        if np.any(t <= 0):
            raise ValueError("All layer thickness values must be positive.")
        if np.any(v <= 0):
            raise ValueError("All velocity values must be positive.")
        if np.any(rho <= 0):
            raise ValueError("All density values must be positive.")

        return cls(thicknesses=t, velocities=v, densities=rho)


@dataclass
class Pulse:
    dominant_frequency: float
    phase: float = 0.0

    def ricker(self, t: np.ndarray) -> np.ndarray:
        """Ricker wavelet centered around zero time."""
        f = self.dominant_frequency
        if f <= 0:
            raise ValueError("dominant_frequency must be positive")
        pi2 = (np.pi * f * (t - self.phase)) ** 2
        return (1.0 - 2.0 * pi2) * np.exp(-pi2)


def interface_reflectivities(model: LayerModel) -> np.ndarray:
    """Acoustic normal-incidence reflection coefficients at each interface."""
    z = model.velocities * model.densities
    return (z[1:] - z[:-1]) / (z[1:] + z[:-1])


def interface_depths(model: LayerModel) -> np.ndarray:
    """Depth of interfaces measured from the surface."""
    return np.cumsum(model.thicknesses)[:-1]


def vertical_twtt_to_interfaces(model: LayerModel) -> np.ndarray:
    """Two-way vertical travel time to each interface."""
    twtt = []
    cumulative = 0.0
    for i in range(len(model.thicknesses) - 1):
        cumulative += 2.0 * model.thicknesses[i] / model.velocities[i]
        twtt.append(cumulative)
    return np.asarray(twtt, dtype=float)


def synthesize_gather(
    model: LayerModel,
    pulse: Pulse,
    receivers: Iterable[float],
    dt: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple CMP-style gather for surface source/receivers.

    Parameters
    ----------
    receivers:
        Receiver offsets (m) from the source at x=0.
    dt, tmax:
        Sampling interval and recording duration in seconds.

    Returns
    -------
    t: (nt,) ndarray
        Time axis.
    data: (n_receivers, nt) ndarray
        Synthetic amplitudes.
    """
    if dt <= 0 or tmax <= 0:
        raise ValueError("dt and tmax must be positive.")

    offsets = np.asarray(list(receivers), dtype=float)
    if offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError("receivers must be a non-empty 1D sequence")
    if np.any(offsets < 0):
        raise ValueError("Receiver offsets cannot be negative")

    nt = int(np.floor(tmax / dt)) + 1
    t = np.arange(nt) * dt

    rc = interface_reflectivities(model)
    z_interfaces = interface_depths(model)
    v_rms = np.sqrt(
        np.cumsum(model.velocities[:-1] ** 2 * model.thicknesses[:-1] / model.velocities[:-1])
        / np.cumsum(model.thicknesses[:-1] / model.velocities[:-1])
    )
    t0 = vertical_twtt_to_interfaces(model)

    data = np.zeros((len(offsets), nt), dtype=float)

    # Sparse reflectivity series using hyperbolic moveout for each interface.
    for irec, x in enumerate(offsets):
        trace = np.zeros(nt, dtype=float)
        for iref in range(len(rc)):
            t_ref = np.sqrt(t0[iref] ** 2 + (x / v_rms[iref]) ** 2)
            idx = int(round(t_ref / dt))
            if 0 <= idx < nt:
                geometric_spreading = 1.0 / np.sqrt(z_interfaces[iref] ** 2 + (x / 2.0) ** 2)
                trace[idx] += rc[iref] * geometric_spreading * z_interfaces[0]

        # Convolve reflectivity with a compact wavelet.
        half_width = max(1, int(0.08 / dt))
        tw = np.arange(-half_width, half_width + 1) * dt
        w = pulse.ricker(tw)
        data[irec] = np.convolve(trace, w, mode="same")

    return t, data
