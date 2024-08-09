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
    def __init__(self, parents, noise_model_pool, add_noise=True):
        self.add_noise = add_noise

    @abc.abstractmethod
    def compute(self, inputs):
        pass

    def add_noise_to_output(self, output):
        if self.add_noise:
            noise = self.noise_model.rvs(size=output.shape)
            return output + noise
        return output


class LinearLeafModel(LeafNodeModel):
    def __init__(self, parents, noise_model_pool, add_noise=True):
        super().__init__(parents, noise_model_pool, add_noise)
        self.coefs = np.random.randn(len(parents))
        self.noise_model = random.choice(noise_model_pool)

    def compute(self, inputs):
        output = np.dot(inputs, self.coefs)
        return self.add_noise_to_output(output)


class PolynomialLeafModel(LeafNodeModel):
    def __init__(self, parents, noise_model_pool, add_noise=True):
        super().__init__(parents, noise_model_pool, add_noise)
        self.coefs = np.random.randn(len(parents))
        self.noise_model = random.choice(noise_model_pool)

    def compute(self, inputs):
        output = np.dot(np.power(inputs, 2), self.coefs)
        return self.add_noise_to_output(output)


class ExponentialLeafModel(LeafNodeModel):
    def __init__(self, parents, noise_model_pool, add_noise=True):
        super().__init__(parents, noise_model_pool, add_noise)
        self.coefs = np.random.randn(len(parents))
        self.noise_model = random.choice(noise_model_pool)

    def compute(self, inputs):
        clamped_inputs = np.clip(inputs, -100, 100)
        output = np.dot(np.exp(clamped_inputs), self.coefs)
        return self.add_noise_to_output(output)


class LogarithmicLeafModel(LeafNodeModel):
    def __init__(self, parents, noise_model_pool, add_noise=True):
        super().__init__(parents, noise_model_pool, add_noise)
        self.coefs = np.random.randn(len(parents))
        self.noise_model = random.choice(noise_model_pool)

    def compute(self, inputs):
        positive_inputs = np.where(inputs > 0, inputs, 1e-9)
        output = np.dot(np.log(positive_inputs), self.coefs)
        return self.add_noise_to_output(output)


if TORCH_AVAILABLE:

    class NeuralNetworkLeafModel(LeafNodeModel):
        def __init__(self, parents, noise_model_pool, add_noise=True):
            super().__init__(parents, noise_model_pool, add_noise)
            input_dim = len(parents)
            self.model = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Linear(input_dim * 2, 1),
            )
            self.noise_model = random.choice(noise_model_pool)

        def compute(self, inputs):
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            output_tensor = self.model(inputs_tensor).detach().numpy().flatten()
            return self.add_noise_to_output(output_tensor)
