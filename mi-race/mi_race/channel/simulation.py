"""Exact SSA simulator for 1D diffusion with time-dependent input.

Only the schedule-driven variant is exposed: ``simulate_ssa_with_schedule``
runs an exact stochastic-simulation-algorithm trajectory on an L-compartment
lattice and accepts a list of (time, amount) release events that are
injected into compartment 0 at the specified times.

The legacy single-pulse / RDME / comparison helpers used by the original
``modular-system/main.py`` demo are intentionally not ported — see
``modular-system/`` on disk if that workflow is ever needed again.
"""
import numpy as np


def diffusion_jump_rate(D: float, S: float) -> float:
    """
    Convert diffusion coefficient to compartment jump rate.

    Parameters
    ----------
    D : float
        Diffusion coefficient [micron^2 / s]
    S : float
        Compartment size [micron]

    Returns
    -------
    float
        Jump rate between neighboring compartments [1 / s]
    """
    return D / (S ** 2)


def simulate_ssa_with_schedule(
    release_schedule: list[tuple[float, int]],
    L: int,
    S: float,
    D: float,
    dt: float,
    T: float,
    rng: np.random.Generator,
    absorbing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SSA on a 1D lattice with TIME-DEPENDENT input.

    ``release_schedule`` is a list of ``(t_i, a_i)`` pairs. At each ``t_i``,
    ``a_i`` molecules are injected into compartment 0 (additively — multiple
    pulses superpose in the same tube). The schedule does not need to be
    sorted. Times outside ``[0, T]`` are dropped with a warning.

    Boundaries:
      - The source end (compartment 0) always reflects.
      - The far end (compartment ``L-1``) **reflects** by default, or **absorbs**
        when ``absorbing=True`` — a molecule at the far end can leave the system
        at the diffusion rate, so total mass drains over time and each symbol's
        signal returns toward zero (it "finishes").

    Returns ``(times, X)``: ``times`` is a length ``n_steps + 1`` linspace from
    0 to ``T``, ``X`` is an ``(n_steps + 1, L)`` integer array of compartment
    populations recorded at each timestep.
    """
    d = diffusion_jump_rate(D, S)
    n_steps = int(np.round(T / dt))
    times = np.linspace(0.0, T, n_steps + 1)

    schedule: list[tuple[float, int]] = []
    for ti, ai in release_schedule:
        ti_f = float(ti)
        ai_i = int(ai)
        if ti_f < 0.0 or ti_f > T:
            print(
                f"[simulate_ssa_with_schedule][WARN] release at t={ti_f} "
                f"outside [0, {T}], ignored"
            )
            continue
        schedule.append((ti_f, ai_i))
    schedule.sort(key=lambda p: p[0])

    x = np.zeros(L, dtype=int)
    X = np.zeros((n_steps + 1, L), dtype=int)

    sched_ptr = 0
    while sched_ptr < len(schedule) and schedule[sched_ptr][0] <= 0.0:
        x[0] += schedule[sched_ptr][1]
        sched_ptr += 1
    X[0] = x.copy()

    t = 0.0
    record_idx = 1

    while t < T:
        while sched_ptr < len(schedule) and schedule[sched_ptr][0] <= t:
            x[0] += schedule[sched_ptr][1]
            sched_ptr += 1

        propensities = []
        events = []
        for i in range(L):
            if x[i] == 0:
                continue
            if i > 0:
                propensities.append(d * x[i])
                events.append((i, i - 1))
            if i < L - 1:
                propensities.append(d * x[i])
                events.append((i, i + 1))
            elif absorbing:
                # far boundary absorbs: the molecule leaves the system (dst = -1)
                propensities.append(d * x[i])
                events.append((i, -1))

        a0 = float(np.sum(propensities))

        if a0 <= 0:
            if sched_ptr < len(schedule):
                t_next_release = schedule[sched_ptr][0]
                while record_idx <= n_steps and times[record_idx] < t_next_release:
                    X[record_idx] = x.copy()
                    record_idx += 1
                t = t_next_release
                continue
            while record_idx <= n_steps:
                X[record_idx] = x.copy()
                record_idx += 1
            break

        tau = rng.exponential(1.0 / a0)
        t_next = t + tau

        if sched_ptr < len(schedule) and schedule[sched_ptr][0] <= t_next:
            t_release = schedule[sched_ptr][0]
            while record_idx <= n_steps and times[record_idx] < t_release:
                X[record_idx] = x.copy()
                record_idx += 1
            t = t_release
            continue

        while record_idx <= n_steps and times[record_idx] <= t_next:
            X[record_idx] = x.copy()
            record_idx += 1

        r = rng.random() * a0
        cumulative = 0.0
        chosen = None
        for a, event in zip(propensities, events):
            cumulative += a
            if r <= cumulative:
                chosen = event
                break

        src, dst = chosen
        x[src] -= 1
        if dst >= 0:
            x[dst] += 1
        # dst == -1 → absorbed at the far boundary; the molecule leaves the system
        t = t_next

    while record_idx <= n_steps:
        X[record_idx] = x.copy()
        record_idx += 1

    return times, X
