from typing import List
from bandit import Arm
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


class ExperimentRunner:
    def __init__(self, n_arms: int, n_runs: int, n_steps: int) -> None:
        """Initialize the experiment runner."""
        self._n_arms = n_arms
        self._n_runs = n_runs
        self._n_steps = n_steps

    def _initialize_arms(self) -> List[Arm]:
        """Initialize the arms for the bandit experiment."""
        return [Arm(mean=np.random.normal(0, 1), std=1) for _ in range(self._n_arms)]

    def run_experiment(
        self, agent_class: "type[Agent]", **agent_kwargs
    ) -> pd.DataFrame:
        """Run the bandit experiment with the given agent and return the results as a pandas DataFrame."""
        data = []
        for run in range(self._n_runs):
            arms = self._initialize_arms()
            agent = agent_class(self._n_arms, **agent_kwargs)

            for step in range(self._n_steps):
                action = agent.select_action()
                reward = arms[action].pull()
                agent.update(action, reward)
                data.append(
                    {
                        "Step": step + 1,
                        "Reward": reward,
                        "Run": run + 1,
                        "Configuration": f"{agent}",
                    }
                )

        return pd.DataFrame(data)
