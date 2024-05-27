import random
import abc


class RootNodeModel(abc.ABC):
    @abc.abstractmethod
    def sample(self, size):
        pass


class RootDistributionModel(RootNodeModel):
    def __init__(self, model_pool=None):
        self.model_pool = model_pool
        self.distribution = random.choice(self.model_pool)
        self.params = self.distribution.stats(moments="mvsk")

    def sample(self, size):
        return self.distribution.rvs(size=size)
