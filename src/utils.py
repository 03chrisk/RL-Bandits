from copy import deepcopy
from typing import Type
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import seaborn as sns
from experiment_runner import ExperimentRunner
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


def softmax_probabilities(q_values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute probabilities using the softmax function.

    Args:
        q_values: Array of q-values.
        temperature: Temperature parameter.

    Returns:
        Array of probabilities.

    """
    exp_values = np.exp(q_values / temperature)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


#################
# For plotting learning curve of greedy and epsilon-greedy


def plot_with_smoothing_and_std(
    df: pd.DataFrame,
    smoothing_window: int = 50,
    num_runs: int = 500,
    fontsize: int = 14,
) -> None:
    """Plot data with smoothing, mean, and standard deviation using Matplotlib.

    Args:
        df: Pandas DataFrame containing columns ["Step", "Reward", "Run", "Configuration"].
        smoothing_window: Size of the window for smoothing rewards and standard deviation.

    """
    # Alternatively, aggregate across "sem" instead of "std" and skip the Sem computation step below
    grouped = (
        df.groupby(["Step", "Configuration"])["Reward"]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped.rename(columns={"mean": "Mean", "std": "Std"}, inplace=True)

    plt.figure(figsize=(12, 6))
    configurations = grouped["Configuration"].unique()
    for config in configurations:
        subset = deepcopy(grouped[grouped["Configuration"] == config])
        subset["Sem"] = subset["Std"] / np.sqrt(num_runs)

        subset["Smoothed Mean"] = subset.groupby("Configuration")["Mean"].transform(
            lambda x: uniform_filter1d(x, size=smoothing_window, mode="nearest")
        )
        subset["Smoothed Sem"] = subset.groupby("Configuration")["Sem"].transform(
            lambda x: uniform_filter1d(x, size=smoothing_window, mode="nearest")
        )

        plt.plot(subset["Step"], subset["Smoothed Mean"], label=f"{config} (±1 SE)")
        plt.fill_between(
            subset["Step"],
            subset["Smoothed Mean"] - subset["Smoothed Sem"],
            subset["Smoothed Mean"] + subset["Smoothed Sem"],
            alpha=0.2,
        )

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(
        "Average Reward Learning Curve with Smoothing and Standard Error",
        fontdict={"fontsize": fontsize},
    )
    plt.xlabel("Steps", fontdict={"fontsize": fontsize})
    plt.ylabel("Average Reward", fontdict={"fontsize": fontsize})
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("data/learning_curve.png")
    plt.show()


def run_and_collect_results(
    runner: ExperimentRunner,
    agent_class: "Type[Agent]",
    hyperparams: list,
    param_name: str,
) -> pd.DataFrame:
    """Run experiments for an agent with different hyperparameter values and collect results.

    Args:
        runner: Runner object.
        agent_class: Agent class.
        hyperparams: List of hyperparameter values.
        param_name: Name of the hyperparameter.

    Returns:
        Pandas DataFrame with results.

    """
    all_dataframes = []
    for param in hyperparams:
        result_data = runner.run_experiment(agent_class, **{param_name: param})
        result_data["Hyperparameter"] = param_name
        result_data["Hyperparameter Value"] = param
        all_dataframes.append(result_data)
    return pd.concat(all_dataframes, ignore_index=True)


#######################
# For Plotting Hyperparmeters performance


def preprocess_hyperparam_data(
    df: pd.DataFrame = None, num_runs: int = 500
) -> pd.DataFrame:
    """Preprocess the data for the hyperparameter experiment."""
    df["Algorithm"] = df["Configuration"].apply(lambda x: x.split(":")[0])
    df.drop(["Configuration", "Hyperparameter"], axis=1, inplace=True)

    df["Reward"] = df.groupby(["Run", "Algorithm", "Hyperparameter Value"])[
        "Reward"
    ].transform(lambda x: uniform_filter1d(x, size=5, mode="nearest"))

    grouped = (
        df.groupby(["Run", "Algorithm", "Hyperparameter Value"])["Reward"]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped = (
        grouped.groupby(["Algorithm", "Hyperparameter Value"])[["mean", "std"]]
        .agg(["mean"])
        .reset_index()
    )

    # Flatten column names after aggregation
    grouped.columns = ["Algorithm", "Hyperparameter Value", "mean", "std"]

    # Compute the standard error of the mean (SEM)
    grouped["sem"] = grouped["std"] / np.sqrt(num_runs)

    return grouped


def plot_hyperparameters(
    df: pd.DataFrame = None, num_runs: int = 500, fontsize: int = 14
) -> None:
    """Plot data with smoothing, mean, and standard deviation using Matplotlib."""
    grouped = preprocess_hyperparam_data(df=df, num_runs=num_runs)
    agent_colors = {
        "Epsilon Greedy": "blue",
        "UCB": "orange",
        "Optimistic Greedy": "green",
        "Softmax": "red",
        "Action Preference": "purple",
    }

    sns.set_palette("Set2")

    unique_hyperparameters = np.unique(grouped["Hyperparameter Value"])
    x_axis_mapping = {value: i for i, value in enumerate(unique_hyperparameters)}

    plt.figure(figsize=(12, 6))

    for algorithm in grouped["Algorithm"].unique():
        subset = deepcopy(grouped[grouped["Algorithm"] == algorithm])
        subset["x_mapped"] = subset["Hyperparameter Value"].map(
            lambda x: x_axis_mapping[x]
        )
        current_color = agent_colors[algorithm]

        plt.plot(
            subset["x_mapped"],
            subset["mean"].to_numpy().flatten(),
            label=algorithm,
            color=current_color,
        )

        plt.errorbar(
            subset["x_mapped"],
            subset["mean"].to_numpy().flatten(),
            yerr=subset["sem"].to_numpy().flatten(),
            fmt="-",
            color=current_color,
            elinewidth=1.5,
            capsize=5,
            capthick=2,
            alpha=0.4,
            markersize=6,
            marker="o",
        )
        x_pos = subset["x_mapped"].iloc[-1]
        y_pos = subset["mean"].iloc[-1]

        plt.text(
            x_pos + 0.1,
            y_pos,
            algorithm,
            fontsize=fontsize,
            verticalalignment="center",
            horizontalalignment="left",
            color=current_color,
        )

    x_axis_labels = [
        "1/128",
        "1/64",
        "1/32",
        "1/16",
        "1/8",
        "1/4",
        "1/2",
        "1",
        "2",
        "4",
    ]
    plt.xticks(
        ticks=range(0, len(x_axis_labels), 1),
        labels=x_axis_labels,
        rotation=45,
        ha="right",
        fontsize=fontsize,
    )

    plt.yticks(fontsize=fontsize)

    # Custom x-axis label with colors for each hyperparameter
    epsilon_color = agent_colors["Epsilon Greedy"]
    alpha_color = agent_colors["Action Preference"]
    c_color = agent_colors["UCB"]
    Q0_color = agent_colors["Optimistic Greedy"]
    tau_color = agent_colors["Softmax"]

    # Set x-axis label colors using LaTeX formatting
    plt.gca().xaxis.set_label_coords(0.5, -0.15)
    plt.annotate(
        r"$\mathbf{\varepsilon}$",
        xy=(0.45, -0.15),
        xycoords="axes fraction",
        color=epsilon_color,
        fontsize=fontsize,
    )
    plt.annotate(
        r"$\mathbf{\alpha}$",
        xy=(0.475, -0.15),
        xycoords="axes fraction",
        color=alpha_color,
        fontsize=fontsize,
    )
    plt.annotate(
        r"$\mathbf{c}$",
        xy=(0.5, -0.15),
        xycoords="axes fraction",
        color=c_color,
        fontsize=fontsize,
    )
    plt.annotate(
        r"$\mathbf{Q_0}$",
        xy=(0.525, -0.15),
        xycoords="axes fraction",
        color=Q0_color,
        fontsize=fontsize,
    )
    plt.annotate(
        r"$\mathbf{\tau}$",
        xy=(0.56, -0.15),
        xycoords="axes fraction",
        color=tau_color,
        fontsize=fontsize,
    )

    (
        plt.ylabel(
            "Average Reward over First 1000 Steps", fontdict={"fontsize": fontsize}
        ),
    )
    plt.xlim(-0.5, 11)
    plt.title(
        "Average Reward for Different Agent Configurations",
        fontdict={"fontsize": fontsize},
    )
    plt.legend(loc="upper left", fontsize=10, frameon=False)
    plt.grid(True, linestyle="--", color="lightgray", alpha=0.9)
    plt.tight_layout()
    plt.savefig("data/hyperparameter_comparison.png")
    plt.show()
