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


####################### Data loading and preprocessing #######################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
dataset = "polblogs"
data = Dataset(root='/tmp/', name=dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


# Split graph into two graphs 
proportion_of_common_links = 0.5
adj1, adj2 = split_dataset(adj, proportion_of_common_links) 


################################ Mahsa attack ###############################
# Perform the attack
# 
modified_adj1 =  adj1.copy()

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


################################# Evaluation ###############################


