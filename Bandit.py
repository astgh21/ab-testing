from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Bandit(ABC):
    """
    Abstract Base Class defining the structure for bandit algorithms.

    Abstract Methods:
        - __init__: Initializes the bandit's true win rate and other parameters.
        - __repr__: String representation of the bandit.
        - pull: Simulates pulling the bandit's arm to get a reward.
        - update: Updates the bandit's parameters based on the received reward.
        - experiment: Runs the bandit experiment.
        - report: Reports results and saves them to CSV files.
    """
    @abstractmethod
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0
        self.r_estimate = 0

    @abstractmethod
    def __repr__(self):
        return f'An Arm with Win Rate {self.p}'

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self, results, algorithm, N):
        """
        Reports experiment results, saves data to CSV, and logs insights.
        """
        if algorithm == 'EpsilonGreedy':
            (cumulative_reward_average, cumulative_reward, cumulative_regret,
             bandits, chosen_bandit, reward, count_suboptimal) = results
        else:
            (cumulative_reward_average, cumulative_reward, cumulative_regret,
             bandits, chosen_bandit, reward) = results

        # Save experiment data to a CSV file
        experiment = pd.DataFrame({
            'Trial': range(N),
            'Bandit': chosen_bandit,
            'Reward': reward,
            'Algorithm': algorithm
        })
        experiment.to_csv(f'{algorithm}_Experiment.csv', index=False)

        # Save final results for each bandit to a CSV file
        final_results = pd.DataFrame({
            'Bandit': [b.p for b in bandits],
            'Estimated_Reward': [b.p_estimate for b in bandits],
            'Pulls': [b.N for b in bandits],
            'Estimated_Regret': [b.r_estimate for b in bandits]
        })
        final_results.to_csv(f'{algorithm}_FinalResults.csv', index=False)

        # Logging insights
        logger.info(f"--- {algorithm} Report ---")
        for b in range(len(bandits)):
            logger.debug(
                f"Bandit with True Win Rate {bandits[b].p:.3f}: "
                f"Pulled {bandits[b].N} times, "
                f"Estimated Reward {bandits[b].p_estimate:.4f}, "
                f"Estimated Regret {bandits[b].r_estimate:.4f}"
            )

        logger.info(f"Cumulative Reward: {sum(reward):.2f}")
        logger.info(f"Cumulative Regret: {cumulative_regret[-1]:.2f}")

        if algorithm == 'EpsilonGreedy':
            suboptimal_rate = count_suboptimal / N
            logger.warning(f"Suboptimal Pull Percentage: {suboptimal_rate:.4%}")

        # Save cumulative rewards and regrets
        summary = pd.DataFrame({
            'Trial': range(N),
            'Cumulative_Reward': cumulative_reward,
            'Cumulative_Regret': cumulative_regret
        })
        summary.to_csv(f'{algorithm}_CumulativeResults.csv', index=False)


class Visualization:
    """
    Provides methods for visualizing the performance of bandit algorithms.

    Methods:
        - plot_rewards: Visualizes average reward convergence over trials (linear and log scale).
        - plot_regrets: Visualizes average regret convergence over trials (linear and log scale).
    """
    @classmethod
    def plot_rewards(cls, rewards, num_trials, optimal_bandit_reward, title="Average Reward Convergence"):
        """
        Plots the average reward convergence of bandit algorithms.
        """
        cumulative_rewards = np.cumsum(rewards)
        average_reward = cumulative_rewards / (np.arange(num_trials) + 1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(average_reward, label="Average Reward")
        ax[0].axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        ax[0].legend()
        ax[0].set_title(f"{title} (Linear Scale)")
        ax[0].set_xlabel("Number of Trials")
        ax[0].set_ylabel("Average Reward")

        ax[1].plot(average_reward, label="Average Reward")
        ax[1].axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        ax[1].legend()
        ax[1].set_title(f"{title} (Log Scale)")
        ax[1].set_xlabel("Number of Trials")
        ax[1].set_ylabel("Average Reward")
        ax[1].set_xscale("log")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_regrets(cls, rewards, num_trials, optimal_bandit_reward, title="Average Regret Convergence"):
        """
        Plots the average regret convergence of bandit algorithms.
        """
        cumulative_rewards = np.cumsum(rewards)
        cumulative_regrets = optimal_bandit_reward * np.arange(1, num_trials + 1) - cumulative_rewards
        average_regrets = cumulative_regrets / (np.arange(num_trials) + 1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(average_regrets, label="Average Regret")
        ax[0].legend()
        ax[0].set_title(f"{title} (Linear Scale)")
        ax[0].set_xlabel("Number of Trials")
        ax[0].set_ylabel("Average Regret")

        ax[1].plot(average_regrets, label="Average Regret")
        ax[1].legend()
        ax[1].set_title(f"{title} (Log Scale)")
        ax[1].set_xlabel("Number of Trials")
        ax[1].set_ylabel("Average Regret")
        ax[1].set_xscale("log")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()


class EpsilonGreedy(Bandit):
    """
    Implements the Epsilon-Greedy algorithm.

    Methods:
        - pull: Simulates pulling the arm with Gaussian noise.
        - update: Updates the bandit's estimated reward using the received reward.
        - experiment: Runs the experiment with dynamic epsilon decay.
        - report: Inherits reporting from the base Bandit class.
    """
    def __init__(self, p):
        super().__init__(p)

    def __repr__(self):
        return f"EpsilonGreedy Bandit with p={self.p}"

    def pull(self):
        return np.random.randn() + self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = (1 - 1.0 / self.N) * self.p_estimate + 1.0 / self.N * x
        self.r_estimate = self.p - self.p_estimate

    @classmethod
    def experiment(cls, BANDIT_REWARDS, N, t=1):
        """
        Conducts the Epsilon-Greedy experiment.
        """
        bandits = [cls(p) for p in BANDIT_REWARDS]
        means = np.array(BANDIT_REWARDS)
        true_best = np.argmax(means)
        count_suboptimal = 0
        EPS = 1 / t

        reward = np.empty(N)
        chosen_bandit = np.empty(N)

        for i in range(N):
            p = np.random.random()
            if p < EPS:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])

            x = bandits[j].pull()
            bandits[j].update(x)

            if j != true_best:
                count_suboptimal += 1

            reward[i] = x
            chosen_bandit[i] = j

            t += 1
            EPS = 1 / t

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        cumulative_regret = (np.arange(1, N + 1) * max(means) - cumulative_reward)

        return cumulative_reward_average, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal

    def report(self, results, algorithm, N):
        super().report(results, algorithm, N)


class ThompsonSampling(Bandit):
    """
    Implements the Thompson Sampling algorithm.

    Methods:
        - pull: Simulates pulling the arm with scaled noise.
        - sample: Generates a sample from the posterior distribution.
        - update: Updates posterior parameters using Bayesian update rules.
        - experiment: Runs the Thompson Sampling experiment.
        - report: Inherits reporting from the base Bandit class.
    """
    def __init__(self, p):
        super().__init__(p)
        self.lambda_ = 1
        self.tau = 1

    def __repr__(self):
        return f"ThompsonSampling Bandit with p={self.p}"

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.p

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate

    def update(self, x):
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate

    @classmethod
    def experiment(cls, BANDIT_REWARDS, N):
        """
        Conducts the Thompson Sampling experiment.
        """
        bandits = [cls(m) for m in BANDIT_REWARDS]
        reward = np.empty(N)
        chosen_bandit = np.empty(N)

        for i in range(N):
            j = np.argmax([b.sample() for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)

            reward[i] = x
            chosen_bandit[i] = j

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        cumulative_regret = (np.arange(1, N + 1) * max([b.p for b in bandits]) - cumulative_reward)

        return cumulative_reward_average, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward

    def report(self, results, algorithm, N):
        super().report(results, algorithm, N)


def comparison(bandit_rewards, num_trials, different_plots=False):
    """
    Compare the performance of Epsilon-Greedy and Thompson Sampling algorithms.

    Parameters:
        - bandit_rewards: List of bandit win rates.
        - num_trials: Number of trials to run.
        - different_plots: Whether to compare algorithms in separate plots.
    """
    logger.info("Starting the comparison between Epsilon Greedy and Thompson Sampling algorithms.")
    _, _, _, _, _, eg_rewards, _ = EpsilonGreedy.experiment(bandit_rewards, num_trials)
    _, _, _, _, _, ts_rewards = ThompsonSampling.experiment(bandit_rewards, num_trials)

    optimal_bandit_reward = max(bandit_rewards)

    Visualization.plot_rewards(eg_rewards, num_trials, optimal_bandit_reward, title="Epsilon Greedy Reward Convergence")
    Visualization.plot_rewards(ts_rewards, num_trials, optimal_bandit_reward, title="Thompson Sampling Reward Convergence")
    Visualization.plot_regrets(eg_rewards, num_trials, optimal_bandit_reward, title="Epsilon Greedy Regret Convergence")
    Visualization.plot_regrets(ts_rewards, num_trials, optimal_bandit_reward, title="Thompson Sampling Regret Convergence")

    if different_plots:
        eg_average_rewards = np.cumsum(eg_rewards) / (np.arange(num_trials) + 1)
        ts_average_rewards = np.cumsum(ts_rewards) / (np.arange(num_trials) + 1)

        plt.plot(eg_average_rewards, label="Epsilon Greedy")
        plt.plot(ts_average_rewards, label="Thompson Sampling")
        plt.axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        plt.legend()
        plt.title("Comparison of Average Reward Convergence")
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.xscale("log")
        plt.show()


def print_suggestions():
    """
    Prints suggestions for improving the code, such as refactoring or adding interactivity.
    """
    suggestions = [
        "Refactor the 'report' and 'experiment' methods to be class methods.",
        "Allow users to set epsilon decay rates or Thompson Sampling precisions interactively.",
        "Create a base class 'BanditAlgorithm' for shared functionality between algorithms."
    ]
    logger.info("Suggestions for Improvement:")
    for suggestion in suggestions:
        logger.info(f"- {suggestion}")

