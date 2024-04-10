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

####################### Data loading and preprocessing #######################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
dataset = "polblogs"
#data = Dataset(root='/tmp/', name=dataset)  : this is for unix-based systems

# Use the current directory for windows
data = Dataset(root='.', name=dataset)
#data = Dataset(root='/tmp/', name=dataset)

adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


# Split graph into two graphs 
proportion_of_common_links = 0.5
adj1, adj2 = split_dataset(adj, proportion_of_common_links) 


################################ Mahsa attack ###############################
# Perform the attack
# 
modified_adj1 =  adj1.copy()

# Create a NetworkX graph from the adjacency matrix
graph = nx.from_scipy_sparse_array(modified_adj1)

# Add labels to the graph
for node_id, label in enumerate(labels):
    graph.nodes[node_id]['label'] = label
# print(f"lenght of the labels : {len(labels)}") 
print(f"graph edges : {graph.number_of_edges()}")


target_node, target_label, budget = backdoor.target(graph)   
non_neighbor_opposit= backdoor.find_non_neighbor_opposit_label(graph, target_node, target_label)
max_same_min_opposit_label_neighbors= backdoor.find_max_same_min_opposit_label_neighbors(graph, non_neighbor_opposit)
nodes_for_attack = backdoor.nodes_for_attack(graph, target_node, max_same_min_opposit_label_neighbors, budget)
attacked_graph = backdoor.insert_edge(graph, target_node, nodes_for_attack)
backdoor.evaluate_graph(attacked_graph, nodes_for_attack, target_node, budget)
modified_adj1 =backdoor.convert(attacked_graph)

print(attacked_graph)

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
accuracies = model.test(idx_test)


#### Mahsa Test after CryptoGraph ####


################################# Evaluation ###############################
#print(f"Test accuracy: {accuracies[0]:.2f}")
#print(f"Test accuracy after attack: {accuracies[1]:.2f}")
#print(f"Test accuracy after defense: {accuracies[2]:.2f}")
#print(f"Test accuracy after attack and defense: {accuracies[3]:.2f}")
