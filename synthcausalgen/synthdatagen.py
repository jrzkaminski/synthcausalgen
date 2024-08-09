import numpy as np
import scipy.stats as stats
import networkx as nx
import random
import pandas as pd

from synthcausalgen.core.node_models.leaf_models import (
    NeuralNetworkLeafModel,
    ExponentialLeafModel,
    PolynomialLeafModel,
    LogarithmicLeafModel,
    LinearLeafModel,
)
from synthcausalgen.core.node_models.root_models import RootDistributionModel

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_continuous_distributions():
    distributions = []
    for dist_name in dir(stats):
        dist = getattr(stats, dist_name)
        if isinstance(dist, stats.rv_continuous):
            # Check if the distribution requires shape parameters
            if dist.shapes is None:
                distributions.append(dist)
    return distributions


_CONTINUOUS_DISTRIBUTIONS = get_continuous_distributions()


class SyntheticDataGenerator:
    def __init__(
        self,
        dag,
        root_model_pool=None,
        leaf_model_pool=None,
        noise_model_pool=None,
        root_params=None,
        add_noise=True,
    ):
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError(
                "The provided graph is not a directed acyclic graph (DAG)."
            )
        self.dag = dag
        self.root_model_pool = (
            root_model_pool
            if root_model_pool is not None
            else _CONTINUOUS_DISTRIBUTIONS
        )
        self.leaf_model_pool = (
            leaf_model_pool
            if leaf_model_pool is not None
            else [
                LinearLeafModel,
                PolynomialLeafModel,
                ExponentialLeafModel,
                LogarithmicLeafModel,
            ]
        )
        if leaf_model_pool is None and TORCH_AVAILABLE:
            self.leaf_model_pool.append(NeuralNetworkLeafModel)
        self.noise_model_pool = (
            noise_model_pool
            if noise_model_pool is not None
            else _CONTINUOUS_DISTRIBUTIONS
        )
        self.root_params = root_params if root_params is not None else {}
        self.add_noise = add_noise
        self.node_descriptions = self._initialize_node_descriptions()

    def _initialize_node_descriptions(self):
        node_descriptions = {}
        for node in self.dag.nodes:
            if list(self.dag.predecessors(node)):
                parents = list(self.dag.predecessors(node))
                model_class = random.choice(self.leaf_model_pool)
                model = model_class(
                    parents, self.noise_model_pool, add_noise=self.add_noise
                )
                node_descriptions[node] = {
                    "type": "leaf",
                    "parents": parents,
                    "model": model,
                    "noise_model": model.noise_model,
                }
            else:
                model = RootDistributionModel(
                    model_pool=self.root_model_pool, params=self.root_params.get(node)
                )
                node_descriptions[node] = {
                    "type": "root",
                    "model": model,
                    "params": model.params,
                }
        return node_descriptions

    def _generate_data_for_node(self, node, data, size):
        if self.node_descriptions[node]["type"] == "root":
            model = self.node_descriptions[node]["model"]
            return model.sample(size)
        else:
            parents = self.node_descriptions[node]["parents"]
            model = self.node_descriptions[node]["model"]
            parent_data = np.column_stack([data[parent] for parent in parents])
            return model.compute(parent_data)

    def get_node_descriptions(self):
        return self.node_descriptions

    def get_graph(self):
        return self.dag

    def get_dataframe(self, size):
        data = {}
        for node in nx.topological_sort(self.dag):
            data[node] = self._generate_data_for_node(node, data, size)
        df = pd.DataFrame(data)
        return df
