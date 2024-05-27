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
        layer_structure,
        root_model_pool=None,
        leaf_model_pool=None,
        noise_model_pool=None,
    ):
        self.layer_structure = layer_structure
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
                NeuralNetworkLeafModel,
            ]
        )
        self.noise_model_pool = (
            noise_model_pool
            if noise_model_pool is not None
            else _CONTINUOUS_DISTRIBUTIONS
        )
        self.graph = nx.DiGraph()
        self.node_descriptions = {}
        self._initialize_layers()

    def _initialize_layers(self):
        for layer, num_nodes in self.layer_structure.items():
            if layer == 0:
                self._initialize_root_features(num_nodes)
            else:
                self._generate_leaf_nodes(layer, num_nodes)

    def _initialize_root_features(self, n_root_features):
        for i in range(n_root_features):
            node_name = f"root_{i}"
            model = RootDistributionModel(model_pool=self.root_model_pool)
            self.graph.add_node(node_name, layer=0)
            self.node_descriptions[node_name] = {
                "type": "root",
                "model": model,
                "params": model.params,
            }

    def _generate_leaf_nodes(self, layer, n_leaf_nodes):
        previous_layers_nodes = [
            node for node, attr in self.graph.nodes(data=True) if attr["layer"] < layer
        ]
        for i in range(n_leaf_nodes):
            node_name = f"leaf_{layer}_{i}"
            num_parents = random.randint(1, len(previous_layers_nodes))
            parents = random.sample(previous_layers_nodes, num_parents)
            self.graph.add_node(node_name, layer=layer)
            for parent in parents:
                self.graph.add_edge(parent, node_name)
            model_class = random.choice(self.leaf_model_pool)
            model = model_class(parents, self.noise_model_pool)
            self.node_descriptions[node_name] = {
                "type": "leaf",
                "parents": parents,
                "model": model,
            }

    def _generate_data_for_node(self, node, data, size):
        if self.node_descriptions[node]["type"] == "root":
            model = self.node_descriptions[node]["model"]
            return model.sample(size)
        else:
            parents = self.node_descriptions[node]["parents"]
            model = self.node_descriptions[node]["model"]
            parent_data = np.column_stack([data[parent] for parent in parents])
            outputs = model.compute(parent_data)
            outputs_with_noise = model.add_noise(outputs)
            return outputs_with_noise

    def get_graph(self):
        return self.graph

    def get_node_descriptions(self):
        return self.node_descriptions

    def get_dataframe(self, size):
        data = {}
        for layer in sorted(self.layer_structure.keys()):
            nodes_in_layer = [
                node
                for node, attr in self.graph.nodes(data=True)
                if attr["layer"] == layer
            ]
            for node in nodes_in_layer:
                data[node] = self._generate_data_for_node(node, data, size)
        columns = {
            node: (
                f"node_l_{self.graph.nodes[node]['layer']}_p_{'_'.join(self.node_descriptions[node]['parents'])}"
                if self.node_descriptions[node]["type"] == "leaf"
                else f"node_l_{self.graph.nodes[node]['layer']}"
            )
            for node in data
        }
        df = pd.DataFrame(data)
        df.rename(columns=columns, inplace=True)
        return df
