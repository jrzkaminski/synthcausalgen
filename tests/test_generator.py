import pytest
import scipy
import pandas as pd

from synthcausalgen.core.node_models.leaf_models import (
    PolynomialLeafModel,
    LinearLeafModel,
)
from synthcausalgen.synthdatagen import SyntheticDataGenerator


def test_initialization():
    layer_structure = {0: 3, 1: 2, 2: 3}
    generator = SyntheticDataGenerator(layer_structure)
    graph = generator.get_graph()
    descriptions = generator.get_node_descriptions()

    total_nodes = sum(layer_structure.values())
    assert len(graph.nodes) == total_nodes
    assert len(descriptions) == total_nodes
    assert descriptions["root_0"]["type"] == "root"
    assert "leaf_1_0" in descriptions
    assert "model" in descriptions["leaf_1_0"]


def test_parents_from_any_layer():
    layer_structure = {0: 3, 1: 2, 2: 3}
    generator = SyntheticDataGenerator(layer_structure)
    graph = generator.get_graph()
    descriptions = generator.get_node_descriptions()

    for node, attr in graph.nodes(data=True):
        if attr["layer"] > 0:
            parents = list(graph.predecessors(node))
            for parent in parents:
                assert graph.nodes[parent]["layer"] < attr["layer"]
                assert parent in descriptions


def test_get_dataframe():
    layer_structure = {0: 3, 1: 2, 2: 3}
    generator = SyntheticDataGenerator(layer_structure)
    df = generator.get_dataframe(size=100)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, sum(layer_structure.values()))


def test_custom_model_pools():
    layer_structure = {0: 3, 1: 2, 2: 3}
    custom_root_model_pool = [
        scipy.stats.norm,
        scipy.stats.laplace,
        scipy.stats.uniform,
        scipy.stats.rayleigh,
    ]
    custom_leaf_model_pool = [LinearLeafModel, PolynomialLeafModel]
    custom_noise_model_pool = [scipy.stats.norm]

    generator = SyntheticDataGenerator(
        layer_structure,
        root_model_pool=custom_root_model_pool,
        leaf_model_pool=custom_leaf_model_pool,
        noise_model_pool=custom_noise_model_pool,
    )
    graph = generator.get_graph()
    descriptions = generator.get_node_descriptions()

    total_nodes = sum(layer_structure.values())
    assert len(graph.nodes) == total_nodes
    assert len(descriptions) == total_nodes
    assert descriptions["root_0"]["type"] == "root"
    assert "leaf_1_0" in descriptions
    assert "model" in descriptions["leaf_1_0"]
    assert isinstance(
        descriptions["leaf_1_0"]["model"], (LinearLeafModel, PolynomialLeafModel)
    )


if __name__ == "__main__":
    pytest.main()
