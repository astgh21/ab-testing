from Bandit import EpsilonGreedy, ThompsonSampling, comparison, print_suggestions
from logs import logger

if __name__ == "__main__":
    BANDIT_REWARDS = [1, 2, 3, 4]
    NUM_TRIALS = 20000

    # Run Epsilon Greedy
    logger.info("Starting Epsilon-Greedy Experiment")
    results_eg = EpsilonGreedy.experiment(BANDIT_REWARDS, NUM_TRIALS)  # Call directly on the class
    eg_bandit = EpsilonGreedy(0)
    eg_bandit.report(results_eg, 'EpsilonGreedy', NUM_TRIALS)  # Ensure report is called

    # Run Thompson Sampling
    logger.info("Starting Thompson Sampling Experiment")
    results_ts = ThompsonSampling.experiment(BANDIT_REWARDS, NUM_TRIALS)  # Call directly on the class
    ts_bandit = ThompsonSampling(0)
    ts_bandit.report(results_ts, 'ThompsonSampling', NUM_TRIALS)  # Ensure report is called

    # Comparison
    logger.info("Starting Comparison of Epsilon-Greedy and Thompson Sampling")
    comparison(BANDIT_REWARDS, NUM_TRIALS, different_plots=True)
    print_suggestions()
