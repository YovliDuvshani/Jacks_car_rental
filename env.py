from functools import lru_cache
from typing import Tuple, Set

import numpy as np

from config import (
    MOVING_COST,
    REWARD_VALUE,
    LAMBDA_RETURN_LOC_1,
    LAMBDA_REQUEST_LOC_1,
    LAMBDA_RETURN_LOC_2,
    LAMBDA_REQUEST_LOC_2,
    MAX_CARS,
    MAX_CAR_SWITCH,
)

from copy import deepcopy

from utils import poisson_distribution, LocationDistributions, Transitions


class Env:
    def __init__(self):
        self._location_distributions_1 = LocationDistributions(
            poisson_distribution(LAMBDA_RETURN_LOC_1),
            poisson_distribution(LAMBDA_REQUEST_LOC_1),
        )
        self._location_distributions_2 = LocationDistributions(
            poisson_distribution(LAMBDA_RETURN_LOC_2),
            poisson_distribution(LAMBDA_REQUEST_LOC_2),
        )

    @lru_cache
    def transitions(self, switched_cars: int):
        (
            transitions_location_1,
            transitions_location_2,
        ) = self._based_transitions

        corrected_transitions_location_1 = Transitions(
            self._apply_action_correction(
                transitions_location_1.probabilities,
                switched_cars,
            ),
            self._apply_action_correction(
                transitions_location_1.rewards,
                switched_cars,
            ),
        )
        corrected_transitions_location_2 = Transitions(
            self._apply_action_correction(
                transitions_location_2.probabilities,
                -switched_cars,
            ),
            self._apply_action_correction(
                transitions_location_2.rewards,
                -switched_cars,
            ),
        )

        transitions = self._combine_locations(
            corrected_transitions_location_1, corrected_transitions_location_2
        )
        transitions.rewards -= MOVING_COST * abs(switched_cars)

        return transitions

    @staticmethod
    def _apply_action_correction(array: np.array, switched_cars: int):
        copied_array = deepcopy(array)
        if switched_cars > 0:
            for number_of_cars in range(MAX_CARS + 1):
                if number_of_cars >= switched_cars:
                    copied_array[number_of_cars] = array[number_of_cars - switched_cars]
                else:
                    copied_array[number_of_cars] = np.zeros(MAX_CARS + 1)
        elif switched_cars < 0:
            for number_of_cars in range(MAX_CARS + 1):
                if number_of_cars - switched_cars <= MAX_CARS:
                    copied_array[number_of_cars] = array[number_of_cars - switched_cars]
                else:
                    copied_array[number_of_cars] = np.zeros(MAX_CARS + 1)
        return copied_array

    @staticmethod
    def _combine_locations(
        transitions_location_1: Transitions,
        transitions_location_2: Transitions,
    ):
        probabilities = np.zeros(
            (MAX_CARS + 1, MAX_CARS + 1, MAX_CARS + 1, MAX_CARS + 1)
        )
        rewards = np.zeros((MAX_CARS + 1, MAX_CARS + 1, MAX_CARS + 1, MAX_CARS + 1))
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                for k in range(MAX_CARS + 1):
                    for l in range(MAX_CARS + 1):
                        probabilities[i, j, k, l] = (
                            transitions_location_1.probabilities[i, j]
                            * transitions_location_2.probabilities[k, l]
                        )
                        rewards[i, j, k, l] = (
                            transitions_location_1.rewards[i, j]
                            + transitions_location_2.rewards[k, l]
                        )
        return Transitions(probabilities, rewards)

    @property
    def _based_transitions(self):
        return (
            self._transitions_one_location(self._location_distributions_1),
            self._transitions_one_location(self._location_distributions_2),
        )

    def _transitions_one_location(
        self, location_distributions: LocationDistributions
    ) -> Transitions:
        probabilities = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        rewards = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

        for number_of_cars in range(MAX_CARS + 1):
            for next_number_of_cars in range(MAX_CARS + 1):
                probability, reward = self._transition_one_location(
                    number_of_cars,
                    next_number_of_cars,
                    location_distributions.returns,
                    location_distributions.requests,
                )
                probabilities[number_of_cars, next_number_of_cars] = probability
                rewards[number_of_cars, next_number_of_cars] = reward
        return Transitions(probabilities, rewards)

    @staticmethod
    def _transition_one_location(
        number_of_cars,
        next_number_of_cars,
        distribution_returned_cars,
        distribution_requested_cars,
    ) -> Tuple[float, float]:
        car_diff = next_number_of_cars - number_of_cars
        expected_partial_reward = 0
        probability = 0

        for number_of_returned_cars in range(MAX_CARS + 1):
            for number_of_requested_car in range(number_of_cars + 1):
                if (car_diff == number_of_returned_cars - number_of_requested_car) or (
                    next_number_of_cars == MAX_CARS
                    and car_diff <= number_of_returned_cars - number_of_requested_car
                ):
                    if number_of_requested_car == number_of_cars:
                        partial_probability = distribution_returned_cars[
                            number_of_returned_cars
                        ] * sum(distribution_requested_cars[number_of_requested_car:])
                    else:
                        partial_probability = (
                            distribution_returned_cars[number_of_returned_cars]
                            * distribution_requested_cars[number_of_requested_car]
                        )
                    probability += partial_probability
                    expected_partial_reward += (
                        number_of_requested_car * partial_probability * REWARD_VALUE
                    )

        return probability, expected_partial_reward / probability

    @staticmethod
    def possible_actions(number_of_cars_loc_1, number_of_cars_loc_2) -> Set[int]:
        negative_possible_actions = {
            -i
            for i in range(
                min(
                    MAX_CAR_SWITCH,
                    MAX_CARS - number_of_cars_loc_1,
                    number_of_cars_loc_2,
                )
                + 1
            )
        }
        positive_possible_actions = {
            i
            for i in range(
                min(
                    MAX_CAR_SWITCH,
                    MAX_CARS - number_of_cars_loc_2,
                    number_of_cars_loc_1,
                )
                + 1
            )
        }
        return positive_possible_actions.union(negative_possible_actions)
