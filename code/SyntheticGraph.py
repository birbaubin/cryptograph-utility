import networkx as nx
import torch
import scipy.sparse as sp
from torch.distributions import Normal


class SyntheticGraph:

    def __init__(self, number_of_nodes, number_of_features):
        self.number_of_nodes = number_of_nodes
        self.number_of_features = number_of_features
    def generate(self):

        graph = nx.barabasi_albert_graph(self.number_of_nodes, 10)


        print(graph)
        np_graph = nx.to_numpy_array(graph).astype(float)
        ts_graph = torch.FloatTensor(np_graph)

        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        features = m.sample(sample_shape=(self.number_of_nodes, self.number_of_features)).reshape([self.number_of_nodes, self.number_of_features])
        # features = torch.rand((self.number_of_nodes, self.number_of_features))

        number_of_reps = 3

        for rep in range(number_of_reps):
            result_features = features.clone()

            for node in graph.nodes():
                for node_neighbor in graph.neighbors(node):
                    # result_features[node] = torch.logical_xor(initial_features[node], initial_features[node_neighbor])
                    result_features[node] = result_features[node] + features[node_neighbor]

            features = torch.nn.functional.normalize(result_features)


        labels = torch.zeros(self.number_of_nodes)
        for node in graph.nodes():
            # labels[node] = features[node].mean() > 0
            labels[node] = features[node].mean() > torch.median(features)
            # print(label)

        number_of_ones = labels.sum()
        print("Number of nodes in class 1 : ", number_of_ones)
        features = features.float()
        labels = labels.long()

        return sp.lil_matrix(np_graph), sp.lil_matrix(features.numpy()), labels.numpy()


#     def test():
#         gcn = GCN(nfeat=features.shape[1],
#                   nhid=16,
#                   nclass=2,
#                   device="cpu",
#                   dropout=0.5,
#                   lr=0.01,
#                   weight_decay=5e-4,
#                   with_relu=True,
#                   with_bias=True
#                   )
#
#         gcn.fit(features, adj, labels, idx_train, idx_val=idx_val, train_iters=200)
#         gcn.eval()
#         return gcn.test(idx_test)
#
#
# adj, features, labels = generate_graph()
#
# print(adj.shape)
#
# idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0], stratify=labels)
#
# test()
