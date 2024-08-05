# tests/test_generator.py

import pytest
import pandas as pd
import networkx as nx
import scipy

from synthcausalgen.core import RandomDAGGenerator
from synthcausalgen.core.node_models.leaf_models import PolynomialLeafModel, NeuralNetworkLeafModel, LinearLeafModel
from synthcausalgen.synthdatagen import SyntheticDataGenerator


def test_initialization():
    dag = RandomDAGGenerator(8, 2, 3, 3, 0.3).generate()
    generator = SyntheticDataGenerator(dag)
    graph = generator.get_graph()
    descriptions = generator.get_node_descriptions()

    total_nodes = 8
    assert len(graph.nodes) == total_nodes
    assert len(descriptions) == total_nodes
    for node in graph.nodes:
        if list(graph.predecessors(node)):
            assert descriptions[node]['type'] == 'leaf'
            assert 'model' in descriptions[node]
            assert 'noise_model' in descriptions[node]
        else:
            assert descriptions[node]['type'] == 'root'


def test_parents_from_any_layer():
    dag = RandomDAGGenerator(8, 2, 3, 3, 0.5).generate()
    generator = SyntheticDataGenerator(dag)
    graph = generator.get_graph()
    descriptions = generator.get_node_descriptions()

    for node, attr in graph.nodes(data=True):
        if list(graph.predecessors(node)):
            parents = list(graph.predecessors(node))
            for parent in parents:
                assert descriptions[parent]


def test_get_dataframe():
    dag = RandomDAGGenerator(8, 2, 3, 3, 0.5).generate()
    generator = SyntheticDataGenerator(dag)
    df = generator.get_dataframe(size=100)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, 8)
    assert all(col.startswith('feature_') for col in df.columns)


def test_custom_model_pools():
    dag = RandomDAGGenerator(8, 2, 3, 3, 0.5).generate()
    custom_root_model_pool = [
        scipy.stats.norm,
        scipy.stats.laplace,
        scipy.stats.uniform,
        scipy.stats.rayleigh,
    ]
    custom_leaf_model_pool = [LinearLeafModel, PolynomialLeafModel]
    custom_noise_model_pool = [scipy.stats.norm]

    generator = SyntheticDataGenerator(
        dag,
        root_model_pool=custom_root_model_pool,
        leaf_model_pool=custom_leaf_model_pool,
        noise_model_pool=custom_noise_model_pool
    )
    graph = generator.get_graph()
    descriptions = generator.get_node_descriptions()

    total_nodes = 8
    assert len(graph.nodes) == total_nodes
    assert len(descriptions) == total_nodes
    for node in graph.nodes:
        if list(graph.predecessors(node)):
            assert descriptions[node]['type'] == 'leaf'
            assert 'model' in descriptions[node]
            assert 'noise_model' in descriptions[node]
            assert isinstance(descriptions[node]['model'], (LinearLeafModel, PolynomialLeafModel))
            assert all(not isinstance(descriptions[node]['model'], NeuralNetworkLeafModel) for node in descriptions if
                       'leaf' in node)


def test_custom_root_params():
    dag = RandomDAGGenerator(8, 2, 3, 3, 0.5).generate()
    root_params = {
        "feature_0": {"loc": 0, "scale": 1},
        "feature_1": {"loc": 5, "scale": 2},
        "feature_2": {"loc": 10, "scale": 3},
    }
    generator = SyntheticDataGenerator(dag, root_params=root_params)
    descriptions = generator.get_node_descriptions()

    assert descriptions['feature_0']['params'] == {"loc": 0, "scale": 1}
    assert descriptions['feature_1']['params'] == {"loc": 5, "scale": 2}
    assert descriptions['feature_2']['params'] == {"loc": 10, "scale": 3}

    if __name__ == '__main__':
        pytest.main()
