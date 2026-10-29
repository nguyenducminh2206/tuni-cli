"""SSA channel: molecules diffuse along a tube and are absorbed by the receiver

Position 0...L-2 are compartments; position L-1 is the receiver. 
Each molecule jumps to each neighbour at rate λ = D / h².
Compartment 0 has a reflecting wall on its left, and a jump into the receiver absorbs the molecule
"""

import numpy as np


def jump_rates(counts: list[int], jump_rate: float) -> list[tuple[int, int, float]]:
    """every jump that can happen next, as (from, to, rate)
    counts[i] is the number of molecules in compartment i (i=0...L-2)
    'to' == len(counts) menas a jump into the receiver (absorption)
    """
    jumps = []
    for i, n in enumerate(counts):
        if n == 0:
            continue                            # an empty compartment cannot send anything
        rate = jump_rate * n                    # each of its n molecules jumps at rate λ
        if i > 0: 
            jumps.append((i, i - 1, rate))      # left (comp 0 has a wall)
        jumps.append((i, i + 1, rate))          # right (from L-2: into the receiver)
    return jumps


def next_jump(jumps: list[tuple[int, int, float]], rng: np.random.Generator) -> tuple[float, int, int]:
    """when the next jump happens and which ont it is, as (wait, from, to)
    the wait (secs) is random with mean 1/total rate: the more that can happen, the sooner somthing does.
    Each jump is picked with probability rate / total rate.
    jumps must not be empty (the empty tube is handled in 2d)
    """
    total = sum(rate for _, _, rate in jumps)
    wait = rng.exponential(1 / total)
    dart = rng.random() * total             # a random point on a ruler of length total
    for source, target, rate in jumps:
        if dart < rate:
            return wait, source, target     # the dart landed on this jump's piece
        dart -= rate                        # not this piece: move on to the next one
    return wait, source, target             # rounding left the dart a hair past the end


def simulate(schedule: list[tuple[float, int]], L: int, jump_rate: float, T: float,
             rng: np.random.Generator) -> tuple[list[float], list[int]]:
    """run the SSA from t=0 to t=T, as (absorption times, counts as T)
    schedule is the release schedule of one symbol, from encoder.release_schedule.
    The end T works like one more release, of 0 molecules: the jumps stop there.
    """
    counts = [0] * (L-1)            # comp 0 ... L-2, all empty at the start 
    absorbed = []                   # the time of every absorption, in order
    t = 0.0
    for release_time, amount in schedule + [(T, 0)]:
        while True:                 # jump until the next release
            jumps = jump_rates(counts, jump_rate)
            if not jumps:
                break               # empty tube: skip straight to the release

            wait, source, target = next_jump(jumps, rng)
            if t + wait >= release_time:
                break               # the release comes first: discard this jump

            t += wait
            counts[source] -= 1
            if target == L - 1:
                absorbed.append(t)  # into the receiver
            else:
                counts[target] += 1
        t = release_time
        counts[0] += amount         # the release: molecules enter comp 0
    return absorbed, counts
