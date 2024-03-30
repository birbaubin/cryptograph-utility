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
import numpy as np
from numba import njit


class LocalCNGCN():

    def __init__(self, nfeat, nhid, nclass, dataset, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True,
                 with_bias=True, device='cpu'):

        self.gcn = GCN(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.dataset = dataset

    def fit(self, adj, features, labels, idx_train, threshold, metric="neighbors", object="links", idx_val=None, k=50, train_iters=200,
            initialize=True, verbose=True, **kwargs):

        if object == "links":

            defense_start = time.time()
            if metric in ["jaccard", "cosine", "neighbors", "adamic_adar", "resource_allocation", "ccpa"]:
                modified_adj = self.drop_dissimilar_edges_with_links(adj, threshold,  metric, **kwargs)
            elif metric == "svd":
                modified_adj = self.truncatedSVD(adj, threshold)

            defense_duration = time.time() - defense_start
            # print("duration of defense : ", defense_duration, "s")

            train_start = time.time()
            self.gcn = self.gcn.to(self.gcn.device)
            self.gcn.fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters,
                        initialize=initialize, verbose=verbose)
            train_end = time.time()
            train_duration = train_end - train_start

        return defense_duration, train_duration, modified_adj
            

    def drop_dissimilar_edges_with_links(self, adj, threshold, metric='neighbors', **kwargs):

        print("Droping dissimilar edges using metric : ", metric)

        np_adj = np.asarray(adj.todense()).astype(int)

        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        # preprocessing based on metric
        edges = np.array(np_adj.nonzero()).T

        removed_cnt = 0


        for edge in edges:
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue

            if metric == 'neighbors':
                score = self._cn_similarity(np_adj[n1], np_adj[n2])

            if metric == 'cosine':
                score = self._cosine_similarity(np_adj[n1], np_adj[n2])

            if metric == 'jaccard':
                score = self._jaccard_similarity(np_adj[n1], np_adj[n2])

            if metric == "adamic_adar":
                score = self._adamic_similarity(np_adj, n1, n2)

            if metric == "resource_allocation":
                score = self._resource_allocation_similarity(np_adj, n1, n2)

            if metric == "ccpa":
                score = self._ccpa_similarity(np_adj, n1, n2, kwargs['alpha'])


            if metric == "neighbors":
                if score <= threshold:
                    if adj[n1, n2] == 1:
                        adj[n1, n2] = 0
                        adj[n2, n1] = 0
                        removed_cnt+= 1

            else:
                if score < threshold:
                    if adj[n1, n2] == 1:
                        adj[n1, n2] = 0
                        adj[n2, n1] = 0
                        removed_cnt+= 1

        print("removed {0} edges".format(removed_cnt))


        return adj
    

    def drop_dissimilar_edges_with_features(self, adj, features, threshold, graph_number, metric='neighbors'):

        print("Droping dissimilar edges using metric : ", metric)

        np_adj = np.asarray(adj.todense()).astype(int)

        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)


        edges = np.array(np_adj.nonzero()).T
        removed_cnt = 0
        for edge in edges:
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue

            if metric == 'neighbors':
                score = self._cn_similarity(np_adj[n1], np_adj[n2])

            if metric == 'cosine':
                score = self._cosine_similarity(np_adj[n1], np_adj[n2])

            if metric == 'jaccard':
                score = self._jaccard_similarity(np_adj[n1], np_adj[n2])

            
                
            if score < threshold:
                    if adj[n1, n2] == 1:
                        adj[n1, n2] = 0
                        adj[n2, n1] = 0
                        removed_cnt+= 1
        print("removed {0} edges in {1} {2}".format(removed_cnt, self.dataset, graph_number))
        
            

        return adj
        

    def _jaccard_similarity(self, a, b):
        intersection =  (a * b).sum()
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
    
    def _ccpa_similarity(self, adj, n1, n2, alpha=0.5):

        #compute dijkstra distance if not already computed
        adj[n1, n2] = 0
        adj[n2, n1] = 0

    
        distance = sp.csgraph.dijkstra(adj, indices=[n1], directed=False, unweighted=True)[0, n2]


        if np.isinf(distance):
            distance = adj.shape[0]

        print(distance)
        score  = alpha * self._cn_similarity(adj[n1], adj[n2]) + (1 - alpha) * (adj.shape[0] / distance)

        # print(score)
        return score
      

    def truncatedSVD(self, data, k=50):
        """Truncated SVD on input data.

        Parameters
        ----------
        data :
            input matrix to be decomposed
        k : int
            number of singular values and vectors to compute.

        Returns
        -------
        numpy.array
            reconstructed matrix.
        """
        print('=== GCN-SVD: rank={} ==='.format(k))
        if sp.issparse(data):
            data = data.asfptype()
            U, S, V = sp.linalg.svds(data, k=k)
            print("rank_after = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(data)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            print("rank_before = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
            print("rank_after = {}".format(len(diag_S.nonzero()[0])))

        return U @ diag_S @ V

    def test(self, idx_test):
        output = self.gcn.test(idx_test)

        return output

    def eval(self):
        self.gcn.eval()
