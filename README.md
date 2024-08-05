# Synthetic Causal Data Generator

[![codecov](https://codecov.io/github/jrzkaminski/synthcausalgen/graph/badge.svg?token=3LI5NF370R)](https://codecov.io/github/jrzkaminski/synthcausalgen)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

My tiny library for creating synthetic tabular data using causal mechanisms on a graph.

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
import pandas as pd
import scipy.stats as stats
from synthcausalgen.core.node_models.leaf_models import (
    ExponentialLeafModel,
    PolynomialLeafModel,
    LogarithmicLeafModel,
)
from synthcausalgen.core.random_dag_generator import RandomDAGGenerator
from synthcausalgen.synthdatagen import SyntheticDataGenerator

# Define the parameters for the RandomDAGGenerator
num_nodes = 8
max_parents = 2
depth = 3
breadth = 3
edge_prob = 0.5

# Generate a random DAG
dag_generator = RandomDAGGenerator(num_nodes, max_parents, depth, breadth, edge_prob)
dag = dag_generator.generate()

# Define custom model pools
custom_root_model_pool = [
    stats.norm,
    stats.laplace,
    stats.t(df=10),
    stats.uniform,
    stats.rayleigh
]

custom_leaf_model_pool = [
    PolynomialLeafModel,
    ExponentialLeafModel,
    LogarithmicLeafModel
]

# If torch is available, include the neural network model
try:
    from synthetic_data_generator.generator import NeuralNetworkLeafModel
    custom_leaf_model_pool.append(NeuralNetworkLeafModel)
except ImportError:
    pass

custom_noise_model_pool = [
    stats.norm,
    stats.uniform,
    stats.expon
]

# Define custom parameters for the root distributions
root_params = {
    "feature_0": {"loc": 0, "scale": 1},
    "feature_1": {"loc": 5, "scale": 2},
    "feature_2": {"loc": 10, "scale": 3},
}

# Initialize the synthetic data generator with custom parameters
generator = SyntheticDataGenerator(
    dag=dag,
    root_model_pool=custom_root_model_pool,
    leaf_model_pool=custom_leaf_model_pool,
    noise_model_pool=custom_noise_model_pool,
    root_params=root_params
)

# Generate a dataframe with synthetic data
df = generator.get_dataframe(size=100)

# Print the first few rows of the dataframe
print(df.head())

# Get the networkx graph
graph = generator.get_graph()

# Print the graph nodes
print(graph.nodes)

# Get the node descriptions
node_descriptions = generator.get_node_descriptions()

# Print the node descriptions
for node, desc in node_descriptions.items():
    print(f"Node: {node}, Description: {desc}")
```

## Contributing

If you suddenly want to contribute to this project, please create a pull request.
For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
