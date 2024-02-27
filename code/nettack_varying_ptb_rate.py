from deeprobust.graph.defense import GCN
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm
from deeprobust.graph.utils import *
from deeprobust.graph.targeted_attack import Nettack

class MultiNettack:
    def __init__(self, adj, features, labels, target_gcn, device, idx_train, idx_test, idx_val):
        self.adj = adj
        self.features = features
        self.target_cn = target_gcn
        self.labels = labels
        self.device = device
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.idx_val = idx_val
        self.surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                        nhid=16,  device=device, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True,
                 with_bias=True)

    def select_nodes(self):
        '''
        selecting nodes as reported in nettack paper:
        (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
        (ii) the 10 nodes with lowest margin (but still correctly classified) and
        (iii) 20 more nodes randomly
        '''

        degrees = self.adj.sum(0).A1
        target_gcn = GCN(nfeat=self.features.shape[1],
                         nhid=16,
                         nclass=self.labels.max().item() + 1,
                         device=self.device
                         )
        target_gcn = target_gcn.to(self.device)
        target_gcn.fit(self.features, self.adj, self.labels, self.idx_train, idx_val=self.idx_val, train_iters=200)
        target_gcn.eval()
        output = target_gcn.predict()

        margin_dict = {}
        for idx in self.idx_test:
            margin = classification_margin(output[idx], self.labels[idx])
            if margin < 0 or degrees[idx] < 1:  # only keep the nodes correctly classified and have at least one link
                continue
            margin_dict[idx] = margin
        sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
        high = [x for x, y in sorted_margins[: 10]]
        low = [x for x, y in sorted_margins[-10:]]
        other = [x for x, y in sorted_margins[10: -10]]
        other = np.random.choice(other, 20, replace=False).tolist()

        return high + low + other

    def attack(self, adj, node_list, perturbation_rate):
        # test on 40 nodes on poisoining attack
        cnt = 0
        degrees = adj.sum(0).A1
        num = len(node_list)
        print('=== [Poisoning] Attacking %s nodes respectively ===' % num)

        modified_adj = adj.copy()
        modified_features = csr_matrix(self.features.copy())
        self.surrogate = self.surrogate.to(self.device)
        self.surrogate.fit(modified_features, modified_adj, self.labels, self.idx_train, self.idx_val)

        for target_node in tqdm(node_list):
            print("Attacking node ", target_node)
            n_perturbations = round(int(degrees[target_node]) * perturbation_rate / 0.2)
            model = Nettack(self.surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=self.device)
            model = model.to(self.device)
            model.attack(modified_features, modified_adj, self.labels, target_node, n_perturbations, verbose=False)
            modified_adj = model.modified_adj
            modified_features = csr_matrix(model.modified_features)

        return modified_adj



