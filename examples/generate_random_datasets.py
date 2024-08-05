import os
import random

import pandas as pd
import scipy.stats as stats
import networkx as nx

from synthcausalgen import SyntheticDataGenerator
from synthcausalgen.core import RandomDAGGenerator
from synthcausalgen.core.node_models import LogarithmicLeafModel, ExponentialLeafModel, PolynomialLeafModel, \
    LinearLeafModel

# Ensure the synthetic_data directory exists
os.makedirs('synthetic_data', exist_ok=True)

# Parameters
num_datasets = 100
num_features = 10
num_samples = 1000


for i in range(num_datasets):
    # Generate a random DAG
    max_parents = random.randint(1, 3)
    depth = random.randint(2, 5)
    breadth = random.randint(3, 4)
    edge_prob = random.uniform(0.3, 0.8)

    dag_generator = RandomDAGGenerator(num_features, max_parents, depth, breadth, edge_prob)
    dag = dag_generator.generate()

    # Define custom model pools for each dataset
    custom_root_model_pool = random.sample([
                                            stats.norm,
                                            stats.laplace,
                                            stats.t(df=10),
                                            stats.uniform,
                                            stats.rayleigh], 5)
    custom_leaf_model_pool = random.sample(
        [
            LinearLeafModel,
            PolynomialLeafModel,
            ExponentialLeafModel,
            LogarithmicLeafModel
        ],
        3
    )

    # If torch is available, include the neural network model
    try:
        from synthetic_data_generator.generator import NeuralNetworkLeafModel

        custom_leaf_model_pool.append(NeuralNetworkLeafModel)
    except ImportError:
        pass

    custom_noise_model_pool = random.sample([
                                            stats.norm,
                                            stats.laplace,
                                            stats.t(df=10),
                                            stats.uniform,
                                            stats.rayleigh], 3)

    # Define random parameters for the root distributions
    root_params = {f"feature_{j}": {"loc": random.uniform(-5, 5), "scale": random.uniform(0.1, 2)} for j in
                   range(num_features)}

    # Initialize the synthetic data generator with custom parameters
    generator = SyntheticDataGenerator(
        dag=dag,
        root_model_pool=custom_root_model_pool,
        leaf_model_pool=custom_leaf_model_pool,
        noise_model_pool=custom_noise_model_pool,
        root_params=root_params
    )

    # Generate a dataframe with synthetic data
    df = generator.get_dataframe(size=num_samples)

    # Save the DAG
    edge_list_path = f'synthetic_data/edge_list_{i}.txt'
    nx.write_edgelist(dag, edge_list_path, data=False)

    # Save the synthetic dataset
    dataset_path = f'synthetic_data/synthetic_dataset_{i}.csv'
    df.to_csv(dataset_path, index=False)

    print(f"Dataset {i + 1}/{num_datasets} saved.")
    print(f"Dataset {i + 1} has {len(dag.nodes)} nodes and {len(df.columns)} features")
