import networkx as nx
import random

import scipy.stats as stats

from synthcausalgen import SyntheticDataGenerator
from synthcausalgen.core.node_models import LinearLeafModel


def create_triangle_dag():
    G = nx.DiGraph()
    G.add_edges_from([('feature_0', 'feature_1'),
                      ('feature_1', 'feature_2'),
                      ('feature_0', 'feature_2')])
    return G

def create_square_dag():
    G = nx.DiGraph()
    G.add_edges_from([('feature_0', 'feature_1'),
                      ('feature_1', 'feature_2'),
                      ('feature_2', 'feature_3'),
                      ('feature_0', 'feature_3')])
    return G

def remove_random_edge(G):
    edge_to_remove = random.choice(list(G.edges()))
    G.remove_edge(*edge_to_remove)
    return G


def generate_datasets(dag_generator,
                      num_samples=1000):
    dag = dag_generator
    generator = SyntheticDataGenerator(dag=dag,
                                       root_model_pool=[stats.norm],
                                       leaf_model_pool=[LinearLeafModel],
                                       noise_model_pool=[stats.norm],
                                       add_noise=False)
    df = generator.get_dataframe(size=num_samples)
    return df

import os

output_dir = "synthetic_data"
os.makedirs(output_dir, exist_ok=True)

num_pairs = 100

for i in range(num_pairs):
    for create_dag in [create_triangle_dag, create_square_dag]:  # Add more as needed
        dag_1 = create_dag()
        dag_2 = remove_random_edge(dag_1.copy())

        df_1 = generate_datasets(dag_1)
        df_2 = generate_datasets(dag_2)

        df_1.to_csv(f"{output_dir}/dataset_{i}_original.csv", index=False)
        df_2.to_csv(f"{output_dir}/dataset_{i}_variant.csv", index=False)

        nx.write_edgelist(dag_1, f"{output_dir}/dag_{i}_original.txt", data=False)
        nx.write_edgelist(dag_2, f"{output_dir}/dag_{i}_variant.txt", data=False)

