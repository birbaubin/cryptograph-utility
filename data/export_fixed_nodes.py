from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *

dataset_name = "citeseer"
data = Dataset(root='/tmp/', name=dataset_name, seed=15, setting="gcn")
groundtruth_adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = get_train_val_test(groundtruth_adj.shape[0], stratify=labels)

train_nodes_file = open(f"{dataset_name}_data/training_nodes.txt",'w+')
test_nodes_file = open(f'{dataset_name}_data/testing_nodes.txt','w+')
eval_nodes_file = open(f'{dataset_name}_data/eval_nodes.txt','w+')

for train_node in idx_train:
    train_nodes_file.write(str(train_node)+"\n")

train_nodes_file.close()
 

for test_node in idx_test:
    test_nodes_file.write(str(test_node)+"\n")

test_nodes_file.close()

for eval_node in idx_val:
    eval_nodes_file.write(str(eval_node)+"\n")
eval_nodes_file.close()



degrees = groundtruth_adj.sum(0).A1
target_gcn = GCN(nfeat=features.shape[1],
                    nhid=16,
                    nclass=labels.max().item() + 1,
                    device="cpu"
                    )
target_gcn = target_gcn.to("cpu")
target_gcn.fit(features, groundtruth_adj, labels, idx_train, idx_val=idx_val, train_iters=200)
target_gcn.eval()
output = target_gcn.predict()

margin_dict = {}
for idx in idx_test:
    margin = classification_margin(output[idx], labels[idx])
    if margin < 0 or degrees[idx] < 15:  # only keep the nodes correctly classified and that have at least 15 link
        continue
    margin_dict[idx] = margin
sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
high = [x for x, y in sorted_margins[: 5]]
low = [x for x, y in sorted_margins[-5:]]
other = [x for x, y in sorted_margins[5: -5]]
other = np.random.choice(other, 10, replace=False).tolist()

attacked_nodes = high + low + other

attacked_nodes_file = open(f'{dataset_name}_data/attacked_nodes.txt', "w+")
for node in attacked_nodes:
    attacked_nodes_file.write(str(node)+"\n")

attacked_nodes_file.close()


idx_train_str = open(f'{dataset_name}_data/training_nodes.txt').readlines()
idx_train =  [eval(i) for i in idx_train_str]

train_nodes_file.close()
eval_nodes_file.close()
test_nodes_file.close()
attacked_nodes_file.close()

print(attacked_nodes)