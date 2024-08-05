import random
import abc


class RootNodeModel(abc.ABC):
    @abc.abstractmethod
    def sample(self, size):
        pass


class RootDistributionModel(RootNodeModel):
    def __init__(self, model_pool=None, params=None):
        self.model_pool = model_pool
        self.distribution = random.choice(self.model_pool)
        self.params = params if params is not None else {}

    def sample(self, size):
        try:
            return self.distribution.rvs(size=size, **self.params)
        except TypeError:
            # Handle distributions that do not accept loc/scale parameters
            return self.distribution.rvs(size=size)
