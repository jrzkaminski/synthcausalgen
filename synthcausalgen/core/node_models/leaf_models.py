import numpy as np
import random
import abc

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LeafNodeModel(abc.ABC):
    @abc.abstractmethod
    def compute(self, inputs):
        pass

    @abc.abstractmethod
    def add_noise(self, output):
        pass


class LinearLeafModel(LeafNodeModel):
    def __init__(self, parents, noise_model_pool):
        self.coefs = np.random.randn(len(parents))
        self.noise_model_pool = noise_model_pool
        self.noise_model = random.choice(self.noise_model_pool)

    def compute(self, inputs):
        return np.dot(inputs, self.coefs)

    def add_noise(self, output):
        noise = self.noise_model.rvs(size=output.shape)
        return output + noise


class PolynomialLeafModel(LeafNodeModel):
    def __init__(self, parents, noise_model_pool):
        self.coefs = np.random.randn(len(parents))
        self.noise_model_pool = noise_model_pool
        self.noise_model = random.choice(self.noise_model_pool)

    def compute(self, inputs):
        return np.dot(np.power(inputs, 2), self.coefs)

    def add_noise(self, output):
        noise = self.noise_model.rvs(size=output.shape)
        return output + noise


class ExponentialLeafModel(LeafNodeModel):
    def __init__(self, parents, noise_model_pool):
        self.coefs = np.random.randn(len(parents))
        self.noise_model_pool = noise_model_pool
        self.noise_model = random.choice(self.noise_model_pool)

    def compute(self, inputs):
        return np.dot(np.exp(inputs), self.coefs)

    def add_noise(self, output):
        noise = self.noise_model.rvs(size=output.shape)
        return output + noise


class LogarithmicLeafModel(LeafNodeModel):
    def __init__(self, parents, noise_model_pool):
        self.coefs = np.random.randn(len(parents))
        self.noise_model_pool = noise_model_pool
        self.noise_model = random.choice(self.noise_model_pool)

    def compute(self, inputs):
        # Ensure inputs are positive
        positive_inputs = np.where(inputs > 0, inputs, 1e-9)
        return np.dot(np.log(positive_inputs), self.coefs)

    def add_noise(self, output):
        noise = self.noise_model.rvs(size=output.shape)
        return output + noise


if TORCH_AVAILABLE:

    class NeuralNetworkLeafModel(LeafNodeModel):
        def __init__(self, parents, noise_model_pool):
            input_dim = len(parents)
            self.model = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Linear(input_dim * 2, 1),
            )
            self.noise_model_pool = noise_model_pool
            self.noise_model = random.choice(self.noise_model_pool)

        def compute(self, inputs):
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            outputs_tensor = self.model(inputs_tensor).detach().numpy().flatten()
            return outputs_tensor

        def add_noise(self, output):
            noise = self.noise_model.rvs(size=output.shape)
            return output + noise
