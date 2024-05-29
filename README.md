# Synthetic Causal Data Generator

[![codecov](https://codecov.io/github/jrzkaminski/synthcausalgen/graph/badge.svg?token=3LI5NF370R)](https://codecov.io/github/jrzkaminski/synthcausalgen)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

My tiny library for creating synthetic data using causal mechanisms on a graph.

## QuickStart

Use the package manager [poetry](https://python-poetry.org/) to install the synthetic data generator.

Clone the repository and run the following command in the root directory of the repository.

```bash
git clone https://github.com/jrzkaminski/synthcausalgen.git
cd synthcausalgen
poetry install
```

If you want to use Leaf model that requires PyTorch, use:

```bash
poetry install --extras "torch"
```

## Usage

Here is a simple example of how to use the synthetic data generator.

```python
from synthcausalgen.synthdatagen import SyntheticDataGenerator
from synthcausalgen.core.node_models.leaf_models import LinearLeafModel, PolynomialLeafModel, ExponentialLeafModel, \
    LogarithmicLeafModel, NeuralNetworkLeafModel
import scipy.stats as stats


# Define custom model pools
custom_root_model_pool = [
    stats.norm,
    stats.laplace,
    stats.t(df=10),
    stats.uniform,
    stats.rayleigh
]

custom_leaf_model_pool = [
    LinearLeafModel,
    PolynomialLeafModel,
    ExponentialLeafModel,
    LogarithmicLeafModel,
    NeuralNetworkLeafModel
]

custom_noise_model_pool = [
    stats.norm,
    stats.uniform,
    stats.expon
]

# Define the layer structure of the graph
layer_structure = {
    0: 3,  # 3 root nodes
    1: 2,  # 2 leaf nodes in the first layer
    2: 3   # 3 leaf nodes in the second layer
}

# Initialize the synthetic data generator
generator = SyntheticDataGenerator(
    layer_structure=layer_structure,
    root_model_pool=custom_root_model_pool,
    leaf_model_pool=custom_leaf_model_pool,
    noise_model_pool=custom_noise_model_pool
)

# Generate a dataframe with synthetic data
df = generator.get_dataframe(size=100)

# Print the first few rows of the dataframe
df.head()

# Tiny quick visualization of the graph
nxgraph = generator.get_graph()

import networkx as nx

nx.draw(nxgraph)

# Get the node descriptions with type, model and parents of each node
generator.get_node_descriptions()
```

## Contributing

If you suddenly want to contribute to this project, please create a pull request.
For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
