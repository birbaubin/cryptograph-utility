import networkx as nx
from deeprobust.graph.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix
#from networkx import convert_matrix
import scipy.sparse
import matplotlib.pyplot as plt
import csv
import pandas as pd #pour lire/ecrire les csv et excel


########################### Attack by inserting edges  ############################
# initializations manually for backdoor attack --- if needed
def target (graph):
    target_node = 678
    budget = 200
    target_label = graph.nodes[target_node]['label']
    print(f"Target node is {target_node} with label {target_label} and budget {budget}")
    return target_node, target_label, budget

#find the neighbor nodes with same labels for target_node 
def find_same_neighbor(graph, target_node):
    neighbor_same = []
    target_label = graph.nodes[target_node]['label']
    for node in graph.nodes():
        if graph.nodes[node]['label'] == target_label and graph.has_edge(target_node, node):
            neighbor_same.append(node)
    return neighbor_same

# find the neighbor nodes with opposit labels for target_node
def find_opposit_neighbor(graph, target_node):
    neighbor_opposit = []
    target_label = graph.nodes[target_node]['label']
    for node in graph.nodes():
        if graph.nodes[node]['label'] != target_label and graph.has_edge(target_node, node):
            neighbor_opposit.append(node)
    return neighbor_opposit

#find the non_neighbor nodes with opposit label for target_node 
def find_non_neighbor_opposit_label(graph, target_node):
    non_neighbor_opposit = []
    target_label = graph.nodes[target_node]['label']
    for node in graph.nodes():
        if graph.nodes[node]['label'] != target_label and not graph.has_edge(target_node, node):
            non_neighbor_opposit.append(node)
    return non_neighbor_opposit

#find the most important opposit lable nodes
# we pass "non_neighbor_opposit" as argument of this ---for inserting edges attack:
# we pass "neighbor_same" as argument of this ---for removing edges attack
def find_max_same_min_opposit_label_neighbors(graph, array_nodes ):
    max_same_min_opposit_label_neighbors = []
    for node in array_nodes:
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
# (insert link between target node and nodes which have more common neighbors with target node)
def nodes_for_attack(graph, target_node, max_same_min_opposit_label_neighbors, budget ):
    nodes_for_attack = []
    for node , p in max_same_min_opposit_label_neighbors:
            common = find_common_neighbors(graph, target_node, node)
            len_common = len(common)
            nodes_for_attack.append((node,p, len_common))
    nodes_for_attack.sort(key=lambda x: (x[2], x[1]), reverse=True) # sort by len_common descending and then by p descending
    return [(node) for node in nodes_for_attack[:budget]]
    

#attack V_0 Mahsa : Insert edges
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
        print(f"Edge insertion done and {insert_edge} edges has been inserted between {target_node} and : {nodes_for_attack}")
        return 
    else:    
        print(f"Edge insertion failed and {insert_edge} edges has been inserted")
    # return insert_edge


def convert(attacked_graph):
    # Convert the graph to a CSR matrix
    adjacency_matrix = nx.adjacency_matrix(attacked_graph)
    # Convert the adjacency matrix to CSR format
    adjacency_matrix_csr = csr_matrix(adjacency_matrix)
    return adjacency_matrix_csr


######################### Attack by removing edges ############################

# Bypass CryptoGraph: find more importent nodes + bypass cryptograph 
# (remove link between target node and nodes which have LESS common neighbors with target node)
def nodes_for_remove(graph, target_node, max_same_min_opposit_label_neighbors, budget ):
    nodes_for_remove = []
    for node , p in max_same_min_opposit_label_neighbors:
            common = find_common_neighbors(graph, target_node, node)
            len_common = len(common)
            nodes_for_remove.append((node,p, len_common))
    nodes_for_remove.sort(key=lambda x: (-x[1], x[2])) # sort by p descending and then by len_common ascending
    return [(node) for node in nodes_for_remove[:budget]]
    

# Attack V_1 Mahsa
# we pass return of max_same_min_opposit_label_neighbors as nodes_for_remove in first step
def remove_edge1(graph, target_node, nodes_for_remove, budget): 
    attacked_graph = graph.copy()
    for node, _  in nodes_for_remove[:budget]:
       attacked_graph.remove_edge(target_node, node)
    return attacked_graph

#seconde step: we pass nodes_for_remove output to remove edges between target node and nodes which have less common neighbors with target node
def remove_edge2(graph, target_node, nodes_for_remove): 
    attacked_graph = graph.copy()
    for node, _ ,_ in nodes_for_remove:
       attacked_graph.remove_edge(target_node, node)
    return attacked_graph

# evaluate function for the attack removing edges
def check_removing(attacked_graph, nodes_for_remove, target_node, budget):
    node1 = target_node
    remove_edge = 0
    for node2, _ , _ in nodes_for_remove:
        if not attacked_graph.has_edge(node1, node2):
            remove_edge += 1
    if remove_edge == budget:
        print(f"Edge removing done and {remove_edge} edges has been removed between {target_node} and : {nodes_for_remove}")
        return 
    else:    
        print(f"Edge removing failed and {remove_edge} edges has been removed")
    # return insert_edge

########################## V1 Mahsa: add and Remove edges ############################
######################### Analyse the nodes ############################
# get a dictionary consist of : for each node (key), has the values of numbers of (same_label, opposite_label)
def count_neighbor_labels(graph, labels):
    count_dict = {} # dictionary type variable with key: node, and value of: (same_label, opposite_label)
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        same_label = sum(labels[neighbor] == labels[node] for neighbor in neighbors)
        opposite_label = len(neighbors) - same_label
        count_dict[node] = (same_label, opposite_label)
    return count_dict

# returns a dict of nodes which have most same label and most opposite labels with the target node
def sort_classly(count_dict, labels, target_node):
    nodes_class_same = {node: counts for node, counts in count_dict.items() if labels[node] == labels[target_node]}
    nodes_class_opp = {node: counts for node, counts in count_dict.items() if labels[node] != labels[target_node]}
    sorted_nodes_same = sorted(nodes_class_same.items(), key=lambda x: x[1][0], reverse=True)
    sorted_nodes_opp = sorted(nodes_class_opp.items(), key=lambda x: x[1][0], reverse=True)
    return sorted_nodes_same, sorted_nodes_opp

# def add_remove(graph, target_node, sorted_nodes_same, sorted_nodes_opp, remove_budget, add_budget):
#     attacked_graph = graph.copy()
#     for i in range(remove_budget):
#         attacked_graph.remove_edge(target_node, sorted_nodes_same[i][0])
#     for i in range(add_budget):
#         attacked_graph.add_edge(target_node, sorted_nodes_opp[i][0])
#     return attacked_graph

def add_remove(graph, target_node, sorted_nodes_same, sorted_nodes_opp, remove_budget, add_budget):
    attacked_graph = graph.copy()
    removed_edges = []
    added_edges = []
    i = 0
    while len(removed_edges) < remove_budget and i < len(sorted_nodes_same):
        edge = (target_node, sorted_nodes_same[i][0])
        if attacked_graph.has_edge(*edge):
            attacked_graph.remove_edge(*edge)
            removed_edges.append(edge)
        else:
            print(f"Nodes {edge} are not neighbors.")
        i += 1
    i = 0
    while len(added_edges) < add_budget and i < len(sorted_nodes_opp):
        edge = (target_node, sorted_nodes_opp[i][0])
        if not attacked_graph.has_edge(*edge):
            attacked_graph.add_edge(*edge)
            added_edges.append(edge)
        else:
            print(f"Nodes {edge} are already neighbors.")
        i += 1
    return attacked_graph, removed_edges, added_edges


def is_neighbor(graph, node1, node2):
    return graph.has_edge(node1, node2) or graph.has_edge(node2, node1)