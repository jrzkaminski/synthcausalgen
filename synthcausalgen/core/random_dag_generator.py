import networkx as nx
import random


class RandomDAGGenerator:
    def __init__(self, num_nodes, max_parents, depth, breadth, edge_prob):
        self.num_nodes = num_nodes
        self.max_parents = max_parents
        self.depth = depth
        self.breadth = breadth
        self.edge_prob = edge_prob
        self.current_node = 0
        self.levels = []

    def generate(self):
        G = nx.DiGraph()
        self.create_nodes()
        self.add_nodes_to_graph(G)
        self.add_edges(G)

        if not nx.is_directed_acyclic_graph(G):
            raise ValueError(
                "Generated graph is not acyclic. Please adjust parameters and try again."
            )

        return G

    def create_nodes(self):
        for level in range(self.depth):
            level_nodes = []
            for _ in range(self.breadth):
                if self.current_node < self.num_nodes:
                    node_name = f"feature_{self.current_node}"
                    level_nodes.append(node_name)
                    self.current_node += 1
            self.levels.append(level_nodes)

    def add_nodes_to_graph(self, G):
        for level_nodes in self.levels:
            for node in level_nodes:
                G.add_node(node)

    def add_edges(self, G):
        for level in range(1, len(self.levels)):
            for node in self.levels[level]:
                self.connect_to_previous_level(G, node, self.levels[level - 1])
                self.connect_to_earlier_levels(G, node, level)

    def connect_to_previous_level(self, G, node, previous_level):
        num_parents = random.randint(1, min(self.max_parents, len(previous_level)))
        parents = random.sample(previous_level, num_parents)
        for parent in parents:
            if random.random() < self.edge_prob:
                G.add_edge(parent, node)

    def connect_to_earlier_levels(self, G, node, current_level):
        for prev_level in range(current_level - 1):
            potential_parents = self.levels[prev_level]
            num_parents = random.randint(
                1, min(self.max_parents, len(potential_parents))
            )
            parents = random.sample(potential_parents, num_parents)
            for parent in parents:
                if random.random() < self.edge_prob:
                    G.add_edge(parent, node)


# # Example usage:
# num_nodes = 20
# max_parents = 3
# depth = 5
# breadth = 5
# edge_prob = 0.3
#
# dag_generator = RandomDAGGenerator(num_nodes, max_parents, depth, breadth, edge_prob)
# dag = dag_generator.generate()
#
# # Visualize the generated DAG
# import matplotlib.pyplot as plt
#
# pos = nx.spring_layout(dag)
# nx.draw(dag, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
# plt.show()
