from numpy.random import normal


class Arm:
    def __init__(self, mean: float, std: float) -> None:
        """Initializes an arm that has a reward following a normal distribution
        Args:
            mean: The mean of the normal distribution
            std: The standard deviation of the normal distribution
        """
        self._mean = mean
        self._std = std

    def pull(self) -> float:
        """Returns a reward following a normal distribution"""
        return normal(loc=self._mean, scale=self._std)
