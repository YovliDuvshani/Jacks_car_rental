from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats import poisson

from config import MAX_CARS


def poisson_distribution(lambda_poisson):
    return [poisson(lambda_poisson).pmf(x) for x in range(MAX_CARS + 1)]


@dataclass
class LocationDistributions:
    returns: List[float]
    requests: List[float]


@dataclass
class Transitions:
    probabilities: np.array
    rewards: np.array
