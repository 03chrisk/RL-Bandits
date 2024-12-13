from abc import ABC, abstractmethod
import numpy as np
from utils import softmax_probabilities


class Agent(ABC):
    """Base class for all agents."""

    @abstractmethod
    def select_action(self) -> int:
        """Select an action based on the agent's policy.

        Returns:
            int: The index of the selected arm.

        """
        pass

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and the reward received.

        Args:
            action: The arm that was pulled.
            reward: The reward received after pulling the arm.

        """
        pass

    @abstractmethod
    def __str__(self) -> None:
        """Return a string representation of the agent"""
        pass


class RandomAgent(Agent):
    def __init__(self, n_arms: int) -> None:
        """Initialize the RandomAgent.

        Args:
            n_arms: The number of arms in the bandit problem.

        """
        self._n_arms = n_arms

    def select_action(self) -> int:
        """Select a random action.

        Returns:
            int: The index of the selected arm.

        """
        return np.random.randint(self._n_arms)

    def update(self, action: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and the reward received.

        Args:
            action: The arm that was pulled.
            reward: The reward received after pulling the arm.

        """
        pass

    def __str__(self) -> None:
        """Return a string representation of the RandomAgent.

        Returns:
            str: The string representation of the RandomAgent.

        """
        return "Random"


class EpsilonGreedyAgent(Agent):
    def __init__(self, n_arms: int, epsilon: float) -> None:
        """Initialize the EpsilonGreedyAgent."""
        self._n_arms = n_arms
        self._epsilon = epsilon
        self._q_values = np.zeros(self._n_arms)
        self._action_counts = np.zeros(self._n_arms)

    def select_action(self) -> int:
        """Select an action based on the agent's policy."""
        if np.random.rand() < self._epsilon:
            return np.random.randint(self._n_arms)
        else:
            return np.argmax(self._q_values)

    def update(self, action: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and the reward received."""
        self._action_counts[action] += 1
        self._q_values[action] = (
            self._q_values[action]
            + (reward - self._q_values[action]) / self._action_counts[action]
        )

    def __str__(self) -> None:
        """Return a string representation of the agent"""
        if self._epsilon == 0:
            return "Greedy"
        else:
            return f"Epsilon Greedy: ε={self._epsilon}"


class UCBAgent(Agent):
    def __init__(self, n_arms: int, c: float = 2) -> None:
        """Initialize the UCBAgent."""
        self._n_arms = n_arms
        self._c = c
        self._q_values = np.zeros(self._n_arms)
        self._action_counts = np.zeros(self._n_arms)

    def select_action(self) -> int:
        """Select an action based on the agent's policy."""
        time = np.sum(self._action_counts) + 1
        uncertainty_term = np.sqrt(
            np.divide(
                np.log(time),
                self._action_counts,
                out=np.full_like(self._action_counts, np.inf),
                where=self._action_counts != 0,
            )
        )

        return np.argmax(self._q_values + self._c * uncertainty_term)

    def update(self, action: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and the reward received."""
        self._action_counts[action] += 1
        self._q_values[action] = (
            self._q_values[action]
            + (reward - self._q_values[action]) / self._action_counts[action]
        )

    def __str__(self) -> None:
        """Return a string representation of the agent"""
        return f"UCB: c={self._c}"


class OptimisticGreedyAgent(Agent):
    def __init__(
        self, n_arms: int, initial_q_value: float = 5.0, alpha: float = 0.1
    ) -> None:
        """Initialize the OptimisticGreedyAgent."""
        self._n_arms = n_arms
        self._initial_q_value = initial_q_value
        self._q_values = np.full(self._n_arms, initial_q_value, dtype=float)
        self._alpha = alpha

    def select_action(self) -> int:
        """Select an action based on the agent's policy."""
        return np.argmax(self._q_values)

    def update(self, action: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and the reward received."""
        self._q_values[action] = self._q_values[action] + self._alpha * (
            reward - self._q_values[action]
        )

    def __str__(self) -> None:
        """Return a string representation of the agent"""
        return f"Optimistic Greedy: α={self._alpha}, Q₀ = {self._initial_q_value}"


class SoftMaxAgent(Agent):
    def __init__(self, n_arms: int, temperature: float = 1.0) -> None:
        """Initialize the SoftMaxAgent."""
        self._n_arms = n_arms
        self._temperature = temperature
        self._q_values = np.zeros(self._n_arms)
        self._action_counts = np.zeros(self._n_arms)

    def select_action(self) -> int:
        """Select an action based on the agent's policy."""
        probabilities = softmax_probabilities(self._q_values, self._temperature)
        return np.random.choice(self._n_arms, p=probabilities)

    def update(self, action: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and the reward received."""
        self._action_counts[action] += 1
        self._q_values[action] = (
            self._q_values[action]
            + (reward - self._q_values[action]) / self._action_counts[action]
        )

    def __str__(self) -> None:
        """Return a string representation of the agent"""
        return f"Softmax: T={self._temperature}"


class ActionPreferenceAgent(Agent):
    def __init__(
        self,
        n_arms: int,
        temperature: float = 1,
        alpha: float = 0.1,
        baseline: bool = True,
    ) -> None:
        """Initialize the ActionPreferenceAgent."""
        self._n_arms = n_arms
        self._temperature = temperature
        self._alpha = alpha
        self._action_preferences = np.zeros(self._n_arms)
        self._baseline = baseline
        self._time = 0
        self._reward_baseline = 0
        self._action_set = np.arange(self._n_arms)

    def select_action(self) -> int:
        """Select an action based on the agent's policy."""
        probabilities = softmax_probabilities(
            self._action_preferences, self._temperature
        )
        return np.random.choice(self._n_arms, p=probabilities)

    def update(self, action_selected: int, reward: float) -> None:
        """Update the agent's knowledge based on the action taken and the reward received."""
        if self._baseline:
            self._time += 1
            self._reward_baseline += (1 / self._time) * (reward - self._reward_baseline)

        mask = np.isin(self._action_set, action_selected)

        self._action_preferences += (
            self._alpha
            * (reward - self._reward_baseline)
            * (
                mask
                - softmax_probabilities(self._action_preferences, self._temperature)
            )
        )

    def __str__(self) -> None:
        """Return a string representation of the agent"""
        return f"Action Preference: T={self._temperature}, α={self._alpha}, baseline={self._baseline}"
