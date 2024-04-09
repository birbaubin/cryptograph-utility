import networkx as nx
from deeprobust.graph.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix
#from networkx import convert_matrix
import scipy.sparse
import matplotlib.pyplot as plt
import csv
import pandas as pd #pour lire/ecrire les csv et excel


# initializations for backdoor attack
def target (graph):
    target_node = 1
    budget = 5
    target_label = graph.nodes[target_node]['label']
    return target_node, target_label, budget

#find the non_neighbor nodes with opposit label for target_node - needed for backdoor attack
def find_non_neighbor_opposit_label(graph, target_node, target_label):
    non_neighbor_opposit = []
    for node in graph.nodes():
        if graph.nodes[node]['label'] != target_label and not graph.has_edge(target_node, node):
            non_neighbor_opposit.append(node)
    return non_neighbor_opposit


#find les most important opposit lable nodes
# we pass "non_neighbor_opposit" as argument of this:
def find_max_same_min_opposit_label_neighbors(graph, non_neighbor_opposit ):
    max_same_min_opposit_label_neighbors = []
    for node in non_neighbor_opposit:
            num_same_neighbors = sum(1 for neighbor in graph.neighbors(node) if graph.nodes[neighbor]['label'] == graph.nodes[node]['label'])
            num_opposit_neighbors = sum(1 for neighbor in graph.neighbors(node) if graph.nodes[neighbor]['label'] != graph.nodes[node]['label'])

            total_neighbors = num_opposit_neighbors + num_same_neighbors
            p = num_same_neighbors/ total_neighbors if total_neighbors != 0 else 0
            # if number of y neighbors are more than x neighbors we accept the node:
            max_same_min_opposit_label_neighbors.append((node,num_same_neighbors,p))  

    max_same_min_opposit_label_neighbors.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return [(node, p) for node, _ ,p in max_same_min_opposit_label_neighbors]

# find common neighbors between 2 nodes
def find_common_neighbors(graph, node1, node2):
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    common_neighbors = neighbors1.intersection(neighbors2)
    return common_neighbors

# Bypass CryptoGraph: find more importent nodes + bypass cryptograph
def nodes_for_attack(graph, target_node, max_same_min_opposit_label_neighbors, budget ):
    nodes_for_attack = []
    for node , p in max_same_min_opposit_label_neighbors:
            common = find_common_neighbors(graph, target_node, node)
            len_common = len(common)
            nodes_for_attack.append((node,p, len_common))
    nodes_for_attack.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return [(node) for node in nodes_for_attack[:budget]]
    

#attack V_0 Mahsa
def insert_edge(graph, target_node, nodes_for_attack): 
    attacked_graph = graph.copy()
    for node, _ , _ in nodes_for_attack:
       attacked_graph.add_edge(target_node, node)
    return attacked_graph

# evaluate function for the attack
def evaluate_graph(attacked_graph, nodes_for_attack, target_node, budget):
    node1 = target_node
    insert_edge = 0
    for node2, _, _ in nodes_for_attack:
        if attacked_graph.has_edge(node1, node2):
            insert_edge += 1
    if insert_edge == budget:
        print(f"Edge insertion is successful and {insert_edge} edges has been inserted between {target_node} and : {nodes_for_attack}")
    else:    
        print(f"Edge insertion is not successful and {insert_edge} edges has been inserted")

# def evaluate_graph_after_cryptograph(attacked_graph, nodes_for_attack, target_node, budget):
#     node1 = target_node
#     insert_edge = 0


def convert(attacked_graph):
    # Convert the graph to a CSR matrix
    adjacency_matrix = nx.adjacency_matrix(attacked_graph)
    # Convert the adjacency matrix to CSR format
    adjacency_matrix_csr = csr_matrix(adjacency_matrix)
    return adjacency_matrix_csr

