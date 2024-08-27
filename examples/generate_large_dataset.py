import os
import networkx as nx
import random
import scipy.stats as stats
from synthcausalgen import SyntheticDataGenerator
from synthcausalgen.core.node_models import LinearLeafModel, LogarithmicLeafModel, ExponentialLeafModel, \
    PolynomialLeafModel
from synthcausalgen.core import RandomDAGGenerator


def remove_random_edge(G):
    edge_to_remove = random.choice(list(G.edges()))
    G.remove_edge(*edge_to_remove)
    return G

def generate_datasets(dag, num_samples=1000, with_noise=False):
    generator = SyntheticDataGenerator(dag,
                                       root_model_pool=[stats.norm,
                                                        stats.rayleigh,
                                                        stats.laplace,
                                                        stats.t(df=10)],
                                       leaf_model_pool=[LinearLeafModel,
                                                        LogarithmicLeafModel,
                                                        ExponentialLeafModel,
                                                        PolynomialLeafModel],
                                       noise_model_pool=[stats.norm],
                                       add_noise=with_noise)
    df = generator.get_dataframe(size=num_samples)
    return df

output_dir = "large_dag_data"
os.makedirs(output_dir, exist_ok=True)

# Generate the initial large DAG
dag_gen = RandomDAGGenerator(num_nodes=100, max_parents=4, depth=10, breadth=10, edge_prob=0.1)
dag = dag_gen.generate()
df = generate_datasets(dag)
df.to_csv(f"{output_dir}/dataset_0.csv", index=False)
nx.write_edgelist(dag, f"{output_dir}/dag_0.txt", data=False)

# Iteratively remove edges and generate datasets
for i in range(1, 101):
    dag = remove_random_edge(dag)
    df = generate_datasets(dag)
    df.to_csv(f"{output_dir}/dataset_{i}.csv", index=False)
    nx.write_edgelist(dag, f"{output_dir}/dag_{i}.txt", data=False)