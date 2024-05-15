import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import FGA
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from tqdm import tqdm
import argparse
from experiments import split_dataset
from DistributedDefense import TwoPartyCNGCN
import networkx as nx
from scipy.sparse import csr_matrix
import Mahsa_backdoor_V0 as backdoor
from sklearn.model_selection import train_test_split

####################### Data loading and preprocessing #######################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
dataset = "polblogs"
# Use the current directory for windows
data = Dataset(root='.', name=dataset)
#data = Dataset(root='/tmp/', name=dataset): this is for unix-based systems

adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

#split idx_test into two parts randomly
test_size = 0.1  # 10% for test_attack, 90% for test_clean
idx_test_attack, idx_test_clean = train_test_split(idx_test, test_size=test_size, random_state=42)

# Split graph into two graphs 
proportion_of_common_links = 0.5
adj1, adj2 = split_dataset(adj, proportion_of_common_links) 

############################ tarin model initially and test accuracy ###########################
# Perform evaluation before attack to find the baseline accuracy

# Train GCN model
model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
            nhid=16, device=device, dropout=0.5)
model = model.to(device)
model.fit(features, adj, labels, idx_train, idx_val=idx_val, patience=30, train_iters=200)
model.eval()
output = model.test(idx_test)
#acc_test = accuracy(output, labels, idx_test)

accuracies = model.test(idx_test) 
print("Test accuracy: ", accuracies)

output_1 = model.test(idx_test_attack)
output_2 = model.test(idx_test_clean)

accuracy_test_attack_1 = model.test(idx_test_attack) 
accuracy_test_clean_1 = model.test(idx_test_clean)

print("Test accuracy on attack set: ", accuracy_test_attack_1)
print("Test accuracy on clean set: ", accuracy_test_clean_1)

################################ Mahsa attack ##############################
# Perform the attack
# version 0: attack on the whole graph - on idx_test_attack nodes

modified_adj1 =  adj1.copy()

# Create a NetworkX graph from the adjacency matrix
graph = nx.from_scipy_sparse_array(modified_adj1)

# Add labels to the graph
for node_id, label in enumerate(labels):
    graph.nodes[node_id]['label'] = label
# print(f"lenght of the labels : {len(labels)}") 
print(f"graph edges : {graph.number_of_edges()}")
print(f"graph nodes : {graph.number_of_nodes()}")
# Set the budget for the attack
budget = 5

#attacked_graph is initially set to graph, and then updated after each attack
# This means that each attack is performed on the graph resulting from the previous attacks.
attacked_graph = graph
for target_node in idx_test_attack:
    non_neighbor_opposit = backdoor.find_non_neighbor_opposit_label(attacked_graph, target_node)
    max_same_min_opposit_label_neighbors = backdoor.find_max_same_min_opposit_label_neighbors(attacked_graph, non_neighbor_opposit)
    nodes_for_attack = backdoor.nodes_for_attack(attacked_graph, target_node, max_same_min_opposit_label_neighbors, budget)
    attacked_graph = backdoor.insert_edge(attacked_graph, target_node, nodes_for_attack)
    backdoor.evaluate_graph(attacked_graph, nodes_for_attack, target_node, budget)

# Convert the graph to a CSR matrix
modified_adj1 =backdoor.convert(attacked_graph)
print(attacked_graph)

################################ evaluation after attack ##############################
# accuracy after attack
model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
            nhid=16, device=device, dropout=0.5)
model = model.to(device)
#data = data.to(device)
model.fit(features, modified_adj1, labels, idx_train, idx_val=idx_val, patience=30, train_iters=200)
model.eval()

output = model.test(idx_test)

#acc_test = accuracy(output, labels, idx_test)

output_1 = model.test(idx_test_attack)
output_2 = model.test(idx_test_clean)

accuracy_test_attack_2 = model.test(idx_test_attack) 
accuracy_test_clean_2 = model.test(idx_test_clean)

print("Test accuracy on attack set: ", accuracy_test_attack_2)
print("Test accuracy on clean set: ", accuracy_test_clean_2)

############################ Crypto'Graph defense ###########################
# Perform Crypto'Graph distributed defense

threshold = 2               # threshold for dropping dissimilar edges
metric = "neighbors"        # metric for dropping dissimilar edges (neighbors, jaccard, cosine)
object = "links"            # object for defense (links, features)

model = TwoPartyCNGCN(dataset=dataset, nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1,
                          device=device)
model.fit(modified_adj1.copy(), adj2.copy(), features, features, labels, idx_train, threshold, metric=metric, object=object,
            train_iters=200, initialize=True, verbose=False, idx_val=idx_val)
model.eval()
accuracies = model.test(idx_test)  #accuracy of the model after the defense on all the test data - (all the nodes)
accuracy_test_attack_3 = model.test(idx_test_attack)
accuracy_test_clean_3 = model.test(idx_test_clean)
print("Test accuracy on attack set: ", accuracy_test_attack_3)
print("Test accuracy on clean set: ", accuracy_test_clean_3)

################################# Evaluation ###############################

