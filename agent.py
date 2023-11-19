import numpy as np
import pandas as pd
from plotly import express as px

from config import GAMMA, EPSILON, MAX_CARS
from env import Env


class Agent:
    def __init__(self, env: Env):
        self.env = env
        self.value_function = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        self.policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

    def evaluate_policy(self):
        loop = True
        while loop:
            delta = 0
            for number_of_cars_loc_1 in range(MAX_CARS + 1):
                for number_of_cars_loc_2 in range(MAX_CARS + 1):
                    current_state_value = self.value_function[
                        number_of_cars_loc_1, number_of_cars_loc_2
                    ]
                    action = self.policy[number_of_cars_loc_1, number_of_cars_loc_2]
                    self.value_function[
                        number_of_cars_loc_1, number_of_cars_loc_2
                    ] = self._evaluate_action_state_value(
                        number_of_cars_loc_1, number_of_cars_loc_2, action
                    )
                    delta = max(
                        delta,
                        abs(
                            current_state_value
                            - self.value_function[
                                number_of_cars_loc_1, number_of_cars_loc_2
                            ]
                        ),
                    )
            print(f"delta: {delta}")
            if delta < EPSILON:
                loop = False

    def improve_policy(self):
        is_stable = False
        while not is_stable:
            is_stable = True
            self.evaluate_policy()
            for number_of_cars_loc_1 in range(MAX_CARS + 1):
                for number_of_cars_loc_2 in range(MAX_CARS + 1):
                    current_action = self.policy[
                        number_of_cars_loc_1, number_of_cars_loc_2
                    ]
                    possible_actions = self.env.possible_actions(
                        number_of_cars_loc_1, number_of_cars_loc_2
                    )
                    action_values = {}
                    for action in possible_actions:
                        action_values[action] = self._evaluate_action_state_value(
                            number_of_cars_loc_1, number_of_cars_loc_2, action
                        )
                    self.policy[number_of_cars_loc_1, number_of_cars_loc_2] = max(
                        action_values, key=action_values.get
                    )
                    if (
                        current_action
                        != self.policy[number_of_cars_loc_1, number_of_cars_loc_2]
                    ):
                        is_stable = False
            self.display_policy()
        self.evaluate_policy()

    def _evaluate_action_state_value(
        self, number_of_cars_loc_1: int, number_of_cars_loc_2: int, action: int
    ):
        transitions = self.env.transitions(action)
        action_state_value = 0
        for new_number_cars_loc_1 in range(MAX_CARS + 1):
            for new_number_cars_loc_2 in range(MAX_CARS + 1):
                action_state_value += transitions.probabilities[
                    number_of_cars_loc_1,
                    new_number_cars_loc_1,
                    number_of_cars_loc_2,
                    new_number_cars_loc_2,
                ] * (
                    transitions.rewards[
                        number_of_cars_loc_1,
                        new_number_cars_loc_1,
                        number_of_cars_loc_2,
                        new_number_cars_loc_2,
                    ]
                    + GAMMA
                    * self.value_function[new_number_cars_loc_1, new_number_cars_loc_2]
                )
        return action_state_value

    def display_policy(self):
        df = pd.melt(
            pd.DataFrame(self.policy).reset_index(),
            id_vars=["index"],
            value_vars=list(range(MAX_CARS + 1)),
        ).rename(
            columns={
                "index": "location_1",
                "variable": "location_2",
                "value": "action",
            }
        )
        fig = px.scatter(
            data_frame=df,
            x="location_2",
            y="location_1",
            color="action",
            text="action",
            opacity=0.5,
        )
        fig.show()
