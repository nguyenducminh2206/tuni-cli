"""encoder: turns a symbol into its release schedule
    (when, and how many molecules, to release into compartment 0)"""

TIME_DECIMALS = 12  # times are rounded to this many decimals to remove float noise


def release_schedule(vector: tuple[int, ...], slot_dt: float) -> list[tuple[float, int]]:
    """the releases of 1 symbol as (time, molecules), in the time order, skipping empty slots
    Slot k releases at t = k * slot_dt. For ex, with slot_dt = 0.1: (50, 0, 0, 20) -> [(0.0, 50), (0.3, 20)]"""
    return [
        (round(k * slot_dt, TIME_DECIMALS), amount) for k, amount in enumerate(vector) if amount > 0
    ]    