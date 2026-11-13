"""encoder search: improve the codebook by moving molecules between slots, scored with the decoders of v1"""

import numpy as np


FLOOR = 0.0025      # smallest prob a decoder may give in the search


def floor_probabilities(probabilities: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    """raise every prob below 'floor' up to 'floor', then rescal each row so it adds up to 1 again
    q_k = max(p_k, floor) / sum_j max(p_j, floor). No symbol is ever called impossible, and q is still
    a probability for every symbol, so the BA bound computed from q stays a true lower bound"""
    raised = np.maximum(probabilities, floor)       # new array
    return raised / raised.sum(axis=1, keepdims=True)