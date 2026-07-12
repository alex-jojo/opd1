import os


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, reward_value=None):
    """Return a fixed reward for baselines that should not use task reward."""
    raw_value = reward_value if reward_value is not None else os.environ.get("CODE_CONSTANT_REWARD_VALUE", "1.0")
    value = float(raw_value)
    return {"score": value, "constant_reward": value}