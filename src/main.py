import numpy as np
import pandas as pd
from agent import (
    EpsilonGreedyAgent,
    UCBAgent,
    OptimisticGreedyAgent,
    SoftMaxAgent,
    ActionPreferenceAgent,
)
from experiment_runner import ExperimentRunner

import os

from utils import (
    plot_with_smoothing_and_std,
    run_and_collect_results,
    plot_hyperparameters,
)

import argparse


def conduct_learning_curve_experiment(rerun_experiment: bool = False) -> None:
    """Conduct the learning curve experiment and save the results to a CSV file."""
    num_arms = 10
    num_runs = 500
    num_steps = 1000

    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "../data/experiment_1_learning_curves.csv")

    if not rerun_experiment and os.path.exists(file_path):
        print("Plotting learning curve from pre-saved data.")
        df = pd.read_csv(file_path)
        plot_with_smoothing_and_std(df, smoothing_window=50, num_runs=num_runs)
        return
    elif not rerun_experiment and not os.path.exists(file_path):
        print("Pre-saved data not found. Rerunning the learning curve experiment.")
    else:
        print("Rerunning the learning curve experiment from scratch.")

    runner = ExperimentRunner(num_arms, num_runs, num_steps)
    results_eps_0 = runner.run_experiment(EpsilonGreedyAgent, epsilon=0.0)
    results_eps_0_1 = runner.run_experiment(EpsilonGreedyAgent, epsilon=0.1)
    df_results = pd.concat([results_eps_0, results_eps_0_1], ignore_index=True)
    df_results.to_csv("data/experiment_1_learning_curves.csv", index=False)
    plot_with_smoothing_and_std(df_results, smoothing_window=50, num_runs=num_runs)


def conduct_hyperparameter_experiment(rerun_experiment: bool = False) -> None:
    """Conduct the hyperparameter experiment and save the results to a CSV file."""
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "../data/experiment_2_hyperparameters.csv")

    if not rerun_experiment and os.path.exists(file_path):
        print("Plotting hyperparameters from pre-saved data.")
        df = pd.read_csv(file_path)
        plot_hyperparameters(df)
        return
    elif not rerun_experiment and not os.path.exists(file_path):
        print("Pre-saved data not found. Rerunning the hyperparameter experiment.")
    else:
        print(
            "Rerunning the hyperparameter experiment from scratch. This might take some time."
        )

    num_arms = 10
    num_runs = 500
    num_steps = 1000

    epsilons = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]
    c_values = [1 / 16, 1 / 4, 1 / 2, 1, 2, 4]
    initial_q_values = [1 / 4, 1 / 2, 1, 2, 4]
    alphas = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
    temperatures = [1 / 16, 1 / 8, 1 / 4, 1 / 2]

    runner = ExperimentRunner(num_arms, num_runs, num_steps)

    epsilon_experiments = run_and_collect_results(
        runner, EpsilonGreedyAgent, epsilons, "epsilon"
    )
    epsilon_experiments.to_csv("data/experiment_2_epsilon.csv", index=False)
    ucb_experiments = run_and_collect_results(runner, UCBAgent, c_values, "c")
    ucb_experiments.to_csv("data/experiment_2_ucb.csv", index=False)
    optimistic_greedy_experiments = run_and_collect_results(
        runner, OptimisticGreedyAgent, initial_q_values, "initial_q_value"
    )
    optimistic_greedy_experiments.to_csv(
        "data/experiment_2_optimistic_greedy.csv", index=False
    )
    softmax_experiments = run_and_collect_results(
        runner, SoftMaxAgent, temperatures, "temperature"
    )
    softmax_experiments.to_csv("data/experiment_2_softmax.csv", index=False)
    action_preference_experiments = run_and_collect_results(
        runner, ActionPreferenceAgent, alphas, "alpha"
    )
    action_preference_experiments.to_csv(
        "data/experiment_2_action_preference.csv", index=False
    )

    all_configurations_combined = pd.concat(
        [
            epsilon_experiments,
            ucb_experiments,
            optimistic_greedy_experiments,
            softmax_experiments,
            action_preference_experiments,
        ],
        ignore_index=True,
    )

    all_configurations_combined.to_csv(
        "data/experiment_2_hyperparameters.csv", index=False
    )

    plot_hyperparameters(all_configurations_combined)


def main() -> None:
    """Main function to conduct the experiments."""
    np.random.seed(1)

    parser = argparse.ArgumentParser(
        description="Run specified machine learning experiments."
    )

    parser.add_argument(
        "--rerun_experiment",
        action="store_true",
        help="Whether to rerun the experiment(s) from scratch (default: False). Note that if this option is selected this may take some time.",
    )

    parser.add_argument(
        "--experiment_choice",
        type=str,
        choices=["learning_curve", "hyperparameter_comparison", "all"],
        default="all",
        help="Which experiment to conduct: 'learning_curve', 'hyperparameter_comparison' or 'all'.",
    )
    args = parser.parse_args()

    if args.experiment_choice == "learning_curve":
        conduct_learning_curve_experiment(rerun_experiment=args.rerun_experiment)
    elif args.experiment_choice == "hyperparameter_comparison":
        conduct_hyperparameter_experiment(rerun_experiment=args.rerun_experiment)
    elif args.experiment_choice == "all":
        conduct_learning_curve_experiment(rerun_experiment=args.rerun_experiment)
        conduct_hyperparameter_experiment(rerun_experiment=args.rerun_experiment)


if __name__ == "__main__":
    main()
