import time

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import lil_matrix
import numpy as np
from numba import njit


class TwoPartyCNGCN():

    def __init__(self, nfeat, nhid, nclass, dataset, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True,
                 with_bias=True, device='cpu'):

        self.device = device
        self.gcn1 = GCN(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.gcn2 = GCN(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.dataset = dataset

    def fit(self, adj1, adj2, features1, features2, labels, idx_train, threshold, metric="neighbors", object="links",
            idx_val=None, train_iters=200,
            initialize=True, verbose=True, **kwargs):

        defense_start = time.time()
        if object == "links":

            if metric in ["neighbors", "jaccard", "cosine", "adamic_adar", "resource_allocation"]:
                self.modified_adj1, self.modified_adj2 = self.drop_dissimilar_edges_with_links(adj1, adj2, threshold, metric)

            elif metric == "svd":
                modified_adj = self.truncatedSVD(adj1, adj2, k=threshold)
                self.k = threshold
                # modified_adj_tensor = utils.sparse_mx_to_torch_sparse_tensor(self.modified_adj)
                # features, modified_adj, labels = utils.to_tensor(features1, modified_adj, labels, device=self.device)
                self.modified_adj1 = self.modified_adj2 = modified_adj

        elif object == "features":
            self.modified_adj1, self.modified_adj2 = self.drop_dissimilar_edges_with_features(adj1, adj2, features1, features2,
                                                                                    threshold, metric)

        defense_duration = time.time() - defense_start

        self.gcn1 = self.gcn1.to(self.device)
        self.gcn2 = self.gcn2.to(self.device)

        training_start1 = time.time()

        self.gcn1.fit(features1, self.modified_adj1, labels, idx_train, idx_val, train_iters=train_iters,
                      initialize=initialize, verbose=verbose)

        training_duration1 = time.time() - training_start1

        training_start2 = time.time()
        self.gcn2.fit(features2, self.modified_adj2, labels, idx_train, idx_val, train_iters=train_iters,
                      initialize=initialize, verbose=verbose)
        training_duration2 = time.time() - training_start2

        return defense_duration, defense_duration, training_duration1, training_duration2

    def drop_dissimilar_edges_with_links(self, adj1, adj2, threshold, metric='neighbors'):

        print("Dropping dissimilar edges using metric : ", metric, " on links")

        np_adj1 = np.asarray(adj1.todense()).astype(int)
        np_adj2 = np.asarray(adj2.todense()).astype(int)
        union_adj = np.bitwise_or(np_adj1, np_adj2)


        if not sp.issparse(adj1):
            adj1 = sp.csr_matrix(adj1)
        if not sp.issparse(adj1):
            adj1 = sp.csr_matrix(adj2)

        edges = np.array(union_adj.nonzero()).T

        removed_cnt1 = 0
        removed_cnt2 = 0

        for edge in edges:
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue

            if metric == 'neighbors':
                score = self._cn_similarity(union_adj[n1], union_adj[n2])

            if metric == 'cosine':
                score = self._cosine_similarity(union_adj[n1], union_adj[n2])

            if metric == 'jaccard':
                score = self._jaccard_similarity(union_adj[n1], union_adj[n2])

            if metric == "adamic_adar":
                score = self._adamic_similarity(union_adj, n1, n2)

            if metric == "resource_allocation":
                score = self._resource_allocation_similarity(union_adj, n1, n2)

            if metric == "neighbors":
                if score <= threshold:
                    if adj1[n1, n2] == 1:
                        adj1[n1, n2] = 0
                        adj1[n2, n1] = 0
                        removed_cnt1 += 1

                    if adj2[n1, n2] == 1:
                        adj2[n1, n2] = 0
                        adj2[n2, n1] = 0
                        removed_cnt2 += 1
            else:
                if score <= threshold:
                    if adj1[n1, n2] == 1:
                        adj1[n1, n2] = 0
                        adj1[n2, n1] = 0
                        removed_cnt1 += 1

                    if adj2[n1, n2] == 1:
                        adj2[n1, n2] = 0
                        adj2[n2, n1] = 0
                        removed_cnt2 += 1



        print("removed {0} edges in {1} 1".format(removed_cnt1, self.dataset))
        print("removed {0} edges in {1} 2".format(removed_cnt2, self.dataset))

        return adj1, adj2

    def drop_dissimilar_edges_with_features(self, adj1, adj2, features1, features2, threshold, metric='neighbors'):

        print("Droping dissimilar edges using metric : ", metric, " on features")

        np_features1 = np.asarray(features1.todense()).astype(int)
        np_features2 = np.asarray(features2.todense()).astype(int)
        union_features = np.bitwise_or(np_features1, np_features2)
        np_adj1 = np.asarray(adj1.todense()).astype(int)
        np_adj2 = np.asarray(adj2.todense()).astype(int)
        union_adj = np.bitwise_or(np_adj1, np_adj2)
        if not sp.issparse(adj1):
            adj1 = sp.csr_matrix(adj1)
        if not sp.issparse(adj1):
            adj1 = sp.csr_matrix(adj2)

        # preprocessing based on metric
        edges = np.array(union_adj.nonzero()).T
        removed_cnt1 = 0
        removed_cnt2 = 0
        for edge in edges:
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue

            if metric == 'neighbors':
                score = self._cn_similarity(union_features[n1], union_features[n2])

            if metric == 'cosine':
                score = self._cosine_similarity(union_features[n1], union_features[n2])

            if metric == 'jaccard':
                score = self._jaccard_similarity(union_features[n1], union_features[n2])

            # print(score)

            if score <= threshold:
                if adj1[n1, n2] == 1:
                    adj1[n1, n2] = 0
                    adj1[n2, n1] = 0
                    removed_cnt1 += 1

                if adj2[n1, n2] == 1:
                    adj2[n1, n2] = 0
                    adj2[n2, n1] = 0
                    removed_cnt2 += 1

        print("removed {0} edges in {1} 1".format(removed_cnt1, self.dataset))
        print("removed {0} edges in {1} 2".format(removed_cnt2, self.dataset))

        return adj1, adj2

    def truncatedSVD(self, adj1, adj2, k=50):

        np_adj1 = np.asarray(adj1.todense()).astype(int)
        np_adj2 = np.asarray(adj2.todense()).astype(int)

        data = np.bitwise_or(np_adj1, np_adj2)

        data = sp.lil_matrix(data)

        data = data.asfptype()
        U, S, V = sp.linalg.svds(data, k=k)

        diag_S = np.diag(S)

        approx_X = U @ diag_S @ V

        # Create a new sparse lil_matrix using the shuffled dense array
        print("rank_after = {}".format(len(S.nonzero()[0])))

        return approx_X

    def _jaccard_similarity(self, a, b):
        intersection = (a * b).sum()
        J = intersection * 1.0 / (a.sum() + b.sum() - intersection)
        return J

    def _cn_similarity(self, a, b):
        intersection = (a * b).sum()
        return intersection

    def _cosine_similarity(self, a, b):
        inner_product = (a * b).sum()
        C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-10)
        return C
    
    def _adamic_similarity(self, adj, n1, n2):
        n1_neighbors = np.nonzero(adj[n1])[0]
        n2_neighbors = np.nonzero(adj[n2])[0]

        score = 0
        for n in n1_neighbors:
            if n in n2_neighbors:
                score += 1 / np.log(adj[n].sum())
        return score
    
    def _resource_allocation_similarity(self, adj, n1, n2):
        n1_neighbors = np.nonzero(adj[n1])[0]
        n2_neighbors = np.nonzero(adj[n2])[0]

        score = 0
        for n in n1_neighbors:
            if n in n2_neighbors:
                score += 1 / adj[n].sum()
        return score

    def test(self, idx_test):
        print("*** " + self.dataset + " 1 ***")
        output1 = self.gcn1.test(idx_test)
        print("*** " + self.dataset + " 2 ***")
        output2 = self.gcn2.test(idx_test)

        return output1, output2
    
  
    

    def eval(self):
        self.gcn1.eval()
        self.gcn2.eval()
