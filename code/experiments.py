import time

import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse as sp
from deeprobust.graph.defense import GCN, GCNJaccard
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, Pyg2Dpr, AmazonPyg
from deeprobust.graph.global_attack import Metattack, Random, DICE, PGDAttack
import argparse
import os
from DistributedDefense import TwoPartyCNGCN
from LocalDefense import LocalCNGCN
from nettack_varying_ptb_rate import MultiNettack
from fgsm_varying_ptb_rate import MultiFGSM
from SyntheticGraph import SyntheticGraph

def convert_duration(seconds):

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 3600) % 60
    return hours, minutes, seconds

def test(adj, idx_test):
    """Evaluation of a 2-layer GNN trained with the provide adjacency matrix. The features and
    labels are from the global context

    Args:
        adj (lil_matrix|csr_matrix|Tensor): adjacency matrix
        idx_test (array_like): list of evaluation nodes

    Returns:
        float : accuracy 
    """
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              device=device,
              dropout=0.5,
              lr=0.01,
              weight_decay=5e-4,
              with_relu=True,
              with_bias=True
              )
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val=idx_val, train_iters=200)
    gcn.eval()
    return gcn.test(idx_test)


def compute_true_positive_rate_and_false_positive_rate(groundtruth_adj, attacked_adj, sanitized_adj):

    groundtruth_adj = groundtruth_adj.toarray()
    attacked_adj = attacked_adj.toarray()
    sanitized_adj = sanitized_adj.toarray()

    TP = np.sum((groundtruth_adj == 0) & (attacked_adj == 1) & (sanitized_adj == 0)) / 2
    FP = np.sum((groundtruth_adj == 1) & (attacked_adj == 1) & (sanitized_adj == 0)) / 2
    TN = np.sum((groundtruth_adj == 1) & (attacked_adj == 1) & (sanitized_adj == 1)) / 2
    FN = np.sum((groundtruth_adj == 0) & (attacked_adj == 1) & (sanitized_adj == 1)) / 2

    print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)

    if TP + FN == 0 or FP + TN == 0:
        return 0.0, 0.0

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return TPR, FPR


def run_metattack(adj, perturbation_rate):

    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val=idx_val)
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                      attack_features=False, device=device, lambda_=1)
    model = model.to(device)
    idx_unlabeled = np.union1d(idx_val, idx_test)
    perturbations = int(perturbation_rate * (adj.sum() // 2))
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)

    return idx_test, lil_matrix(model.modified_adj)


def run_random_attack(adj, perturbation_rate):
  
    perturbations = int(perturbation_rate * (adj.sum() // 2))
    model = Random()
    model.attack(ori_adj=adj, n_perturbations=perturbations, type="add")
    result = lil_matrix(model.modified_adj)
    return idx_test, result


def run_dice_attack(adj, perturbation_rate):

    n_perturbations = int(perturbation_rate * (adj.sum() // 2))
    model = DICE()
    model.attack(adj, labels, n_perturbations)
    modified_adj = model.modified_adj
    result = lil_matrix(modified_adj)

    return idx_test, result


def run_pgd_attack(adj, perturbation_rate):    
    """Run the PGD attack on the provided adjacency matrix, with the specified perturbation rate.

    Args:
        adj (lil_matrix|csr_matrix|Tensor): adjacency matrix
        perturbation_rate (int): perturbation rate

    Returns:
        (array_like, lil_matrix|csr_matrix|Tensor): nodes targeted by the attack, and adjacency matrix after attack
    """

    n_perturbations = int(perturbation_rate * (adj.sum() // 2))
    victim_model = GCN(nfeat=features.shape[1],
                       nhid=16,
                       nclass=labels.max().item() + 1,
                       device=device
                       )
    victim_model = victim_model.to(device)
    victim_model.fit(features, adj, labels, idx_train, idx_val=idx_val)
    model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)

    local_adj, local_features, local_labels = preprocess(adj, features, labels, preprocess_adj=False)

    model = model.to(device)
    model.attack(local_features, local_adj, local_labels, idx_train, n_perturbations=n_perturbations)
    modified_adj = model.modified_adj

    return idx_test, lil_matrix(modified_adj)


def run_nettack(adj, perturbation_rate):
    """Run the nettack attack on the provided adjacency matrix, with the specified perturbation rate.

    Args:
        adj (lil_matrix|csr_matrix|Tensor): adjacency matrix
        perturbation_rate (int): perturbation rate

    Returns:
        (array_like, lil_matrix|csr_matrix|Tensor): nodes targeted by the attack, and adjacency matrix after attack
    """
    attack_model = MultiNettack(adj, features, labels, None, device, idx_train, idx_test, idx_val)
    # attacked_nodes = attack_model.select_nodes()
    modified_adj = attack_model.attack(adj, attacked_nodes, perturbation_rate)

    return attacked_nodes, modified_adj


def run_fgsm(adj, perturbation_rate):
    """Run the IG-FGSM attack on the provided adjacency matrix, with the specified perturbation rate.

    Args:
        adj (lil_matrix|csr_matrix|Tensor): adjacency matrix
        perturbation_rate (int): perturbation rate

    Returns:
        (array_like, lil_matrix|csr_matrix|Tensor): nodes targeted by the attack, and adjacency matrix after attack
    """
      
    attack_model = MultiFGSM(adj, features, labels, None, device, idx_train, idx_test, idx_val)
    modified_adj = attack_model.attack(adj, attacked_nodes, perturbation_rate)
    return attacked_nodes, modified_adj


def run_dist_defense(adj1, adj2, features1, features2, threshold, metric, object, idx_test):
    """Run the distributed defense algorithms on the two graphs

    Args:
        adj1 (lil_matrix|csr_matrix|Tensor): adjacency matrix of the first graph
        adj2 (lil_matrix|csr_matrix|Tensor): adjacency matrix of the second graph
        features1 (lil_matrix|csr_matrix|Tensor): features of the first graph (normally equals the the groundtruth features)
        features2 (lil_matrix|csr_matrix|Tensor): features of the second graph (normally equals the the groundtruth features)
        threshold (float): value of similarity between two nodes under which their link is removed (see paper)
        metric (str): metric used for similarity evaluation (see paper)
        object (_type_): _description_
        idx_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    '''
      Run jaccard, cosine, common neighbors or svd defense on joint graph,
      and remove links in local graph
    '''
    model = TwoPartyCNGCN(dataset=dataset, nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1,
                          device=device)
    times = model.fit(adj1.copy(), adj2.copy(), features1, features2, labels, idx_train, threshold, metric=metric, object=object,
              train_iters=200, initialize=True, verbose=False, idx_val=idx_val)
    model.eval()
    accuracies = model.test(idx_test)

    TP1, FP1 = compute_true_positive_rate_and_false_positive_rate(graph1_adj, adj1, model.modified_adj1)
    TP2, FP2 = compute_true_positive_rate_and_false_positive_rate(graph2_adj, adj2, model.modified_adj2)
    return accuracies[0], accuracies[1], times[0], times[1], times[2], times[3], TP1, TP2, FP1, FP2



def run_local_defense(adj1, adj2, features1, features2, threshold, metric, object, idx_test, **kwargs):
    '''
    Run jaccard, cosine, common neighbors or svd defense on each graph
    individually
    '''

    if metric in ["jaccard", "neighbors", "cosine", "adamic_adar", "resource_allocation", "svd", "ccpa"]:
        if object == "links":
            model1 = LocalCNGCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device,
                                dataset=dataset)
            defense_duration1, training_duration1, modified_adj1 = model1.fit(adj1.copy(), features1, labels, idx_train, threshold, metric=metric, object=object, idx_val=idx_val,
                       train_iters=200, verbose=False, **kwargs)
            output1 = model1.test(idx_test)

            TP1, FP1 = compute_true_positive_rate_and_false_positive_rate(graph1_adj, adj1, modified_adj1)

            model2 = LocalCNGCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device,
                                dataset=dataset)
            defense_duration2, training_duration2, modified_adj2 = model2.fit(adj2.copy(), features2, labels, idx_train, threshold, metric=metric, object=object, idx_val=idx_val,
                       train_iters=200, verbose=False, **kwargs)

            output2 = model2.test(idx_test)
            TP2, FP2 = compute_true_positive_rate_and_false_positive_rate(graph2_adj, adj2, modified_adj2)

        return output1, output2, defense_duration1, defense_duration2, training_duration1, training_duration2, TP1, TP2, FP1, FP2



def run_dist_cn_features_defense(adj1, adj2, threshold, metric):
    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    model = model.to(device)
    model.fit(features, adj1, labels, idx_train, idx_val=idx_val, train_iters=200, threshold=threshold, verbose=False)
    print("*** " + dataset + " 1 ***")
    output1 = model.test(idx_test)

    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    model = model.to(device)
    model.fit(features, adj2, labels, idx_train, idx_val=idx_val, train_iters=200, threshold=threshold, verbose=False)
    print("*** " + dataset + " 2 ***")
    output2 = model.test(idx_test)


# split dataset in two subgraphs
def split_dataset(groundtruth_adj, proportion):

    q1 = 0
    q2 = (1 - proportion) / 2
    q3 = 1 - proportion

    graph1_adj = groundtruth_adj.copy()
    graph2_adj = groundtruth_adj.copy()

    edges = np.array(groundtruth_adj.nonzero()).T

    for edge in edges:
        n1 = edge[0]
        n2 = edge[1]
        if n1 > n2:
            continue
        prob = np.random.random()
        if prob < q2:
            graph2_adj[n1, n2] = 0
            graph2_adj[n2, n1] = 0
        if prob <= q1 or (q3 >= prob >= q2):
            graph1_adj[n1, n2] = 0
            graph1_adj[n2, n1] = 0

    return graph1_adj, graph2_adj


def run_pipeline(attacks, defenses, perturbation_rate_graph1, perturbation_rate_graph2):


    print("\n\n=========== Base graphs ==========")
    print("*** " + dataset + " 1 ***")
    output1 = test(graph1_adj, idx_test)
    print("*** " + dataset + " 2 ***")
    output2 = test(graph2_adj, idx_test)
    logs_of_defenses_with_attacks.write(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(run, None, None, None, None, None, 0, output1, output2, 
                                                                         None, None, None, None, None, None, None,None,None,None))

    for attack in attacks:

        if attack == "metattack": attack_function = run_metattack
        if attack == "random": attack_function = run_random_attack
        if attack == "dice": attack_function = run_dice_attack
        if attack == "pgd": attack_function = run_pgd_attack
        if attack == "nettack": attack_function = run_nettack
        if attack == "fgsm": attack_function = run_fgsm


        print("=========== Attack : ", attack, " Defense: None ==========")
        print("*** " + dataset + " 1, perturbation rate: " + str(perturbation_rate_graph1) + " ***")
        start = time.time()

        if perturbation_rate_graph1 == 0:
            attacked_nodes_graph1 = attacked_nodes
            graph1_attacked = graph1_adj.copy()
        else:
            # print(graph1_adj, perturbation_rate_graph1)
            attacked_nodes_graph1, graph1_attacked = attack_function(graph1_adj.copy(), perturbation_rate_graph1)

        attack_duration1 = time.time() - start
        output1 = test(graph1_attacked, attacked_nodes_graph1)

        print("*** " + dataset + " 2 , perturbation rate: " + str(perturbation_rate_graph2) + " ***")
        start = time.time()
        if perturbation_rate_graph2 == 0:
            attacked_nodes_graph2 = attacked_nodes
            graph2_attacked = graph2_adj.copy()
        else:
            attacked_nodes_graph2, graph2_attacked = attack_function(graph2_adj.copy(), perturbation_rate_graph2)
            
        attack_duration2 = time.time() - start
        output2 = test(graph2_attacked, attacked_nodes_graph2)


        # sp.save_npz("graph1-attacked.npz", graph1_attacked.tocsr())
        # sp.save_npz("graph2-attacked.npz", graph2_attacked.tocsr())

        logs_of_defenses_with_attacks.write(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(run, None, attack, None, None, None, 0,output1,
                                                                 output2, attack_duration1, attack_duration2, None, None, None,None,None,None,None,None))

        features1 = features2 = features

        print("Size of attacked graph-1 : " + str(len(graph1_attacked.nonzero()[0]) / 2))
        print("Size of attacked graph-2 : " + str(len(graph2_attacked.nonzero()[0]) / 2))

        for defense in defenses:
            if defense[2] == "links":
                if defense[1] == "jaccard":
                    thresholds = jaccard_links_thresholds
                elif defense[1] == "neighbors":
                    thresholds = neighbors_links_thresholds
                elif defense[1] == "cosine":
                    thresholds = cosine_links_thresholds
                elif defense[1] == "svd":
                    thresholds = svd_k
                elif defense[1] == "adamic_adar":
                    thresholds = adamic_adar_links_thresholds
                elif defense[1] == "resource_allocation":
                    thresholds = resource_allocation_links_thresholds
                elif defense[1] == "ccpa":
                    thresholds  = neighbors_links_thresholds


            elif defense[2] == "features":
                if defense[1] == "jaccard":
                    thresholds = jaccard_features_thresholds
                elif defense[1] == "neighbors":
                    thresholds = neighbors_features_thresholds
                elif defense[1] == "cosine":
                    thresholds = cosine_features_thresholds

            if defense[0] == "local":
                defense_function = run_local_defense
            else:
                defense_function = run_dist_defense

            for threshold in thresholds:

                if defense[1] == "ccpa":
                    for alpha in np.linspace(0.1, 1.1, 10, endpoint=False):
                        print("=========== Attack : None, Defense " + defense[0] + "-" + defense[1] + "-" + defense[
                            2] + ", Threshold:", threshold, "==========")
                        output = defense_function(graph1_adj.copy(), graph2_adj.copy(), features1.copy(), features2.copy(),
                                                  threshold, metric=defense[1], object=defense[2], idx_test=attacked_nodes_graph1, alpha=alpha)
                        
                        
                        logs_of_defenses_with_attacks.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(run, threshold, None, defense[0], defense[1], defense[2],alpha,
                                                       output[0], output[1], None, None, output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9]))
                        
                        print("=========== Attack : " + attack + ", Defense : " + defense[0] + "-" + defense[1] + "-" + defense[
                            2] + ", Threshold:", threshold, "==========")
                        output = defense_function(graph1_attacked.copy(), graph2_attacked.copy(), features1.copy(),
                                                    features2.copy(),
                                                    threshold, metric=defense[1], object=defense[2],
                                                    idx_test=attacked_nodes_graph2, alpha=alpha)
                        
                        logs_of_defenses_with_attacks.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(run, threshold, None, defense[0], defense[1], defense[2],alpha,
                                                       output[0], output[1], None, None, output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9]))

                else:

                    print("=========== Attack : None, Defense " + defense[0] + "-" + defense[1] + "-" + defense[
                        2] + ", Threshold:", threshold, "==========")
                    
                    output = defense_function(graph1_adj.copy(), graph2_adj.copy(), features1.copy(), features2.copy(),
                                            threshold, metric=defense[1], object=defense[2],
                                            idx_test=attacked_nodes_graph1)
                    
                    logs_of_defenses_with_attacks.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(run, threshold, None, defense[0], defense[1], defense[2],0,
                                                        output[0], output[1], None, None, output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9]))

                    print("=========== Attack : " + attack + ", Defense : " + defense[0] + "-" + defense[1] + "-" + defense[
                        2] + ", Threshold:", threshold, "==========")

                    output = defense_function(graph1_attacked.copy(), graph2_attacked.copy(), features1.copy(),
                                            features2.copy(),
                                            threshold, metric=defense[1], object=defense[2],
                                            idx_test=attacked_nodes_graph2)

                    logs_of_defenses_with_attacks.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(run, threshold, attack, defense[0], defense[1], defense[2],0,
                                                        output[0], output[1], attack_duration1, attack_duration2, output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9]))


project_path = "/Users/aubinbirba/Works/gnn-defense-link-prediction/gnn-attack-defense/"

if __name__ == '__main__':

    exp_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--nbr_exp', type=int, default=5, help='Number of experiments')
    parser.add_argument('--dataset', default='polblogs',
                        choices=["acm", "cora", "citeseer", "pubmed", "polblogs", "flickr", "blogcatalog", "syn"],
                        help='dataset')
    parser.add_argument('--ptb_rate_graph1', type=float, default=0.2, help='pertubation rate graph1')
    parser.add_argument('--ptb_rate_graph2', type=float, default=0.2, help='pertubation rate graph2')
    parser.add_argument('--attack', nargs='+', default=['nettack'],
                        choices=["nettack", "dice", "pgd", "metattack", "fgsm"], help='attacks')
    parser.add_argument('--proportion', type=float, default=0.3, help='Proportion of the groundtruth that is common to the two graphs')
    parser.add_argument('--thresholds', nargs='+', type=int,
                                help='thresholds for local and distributed jaccard and cosine defenses')
    parser.add_argument('--expe_name', type=str,
                                help='experiment name')

    args = parser.parse_args()

    seed = args.seed
    number_of_runs = args.nbr_exp
    dataset = args.dataset
    attacks = args.attack
    perturbation_rate_graph1 = args.ptb_rate_graph1
    perturbation_rate_graph2 = args.ptb_rate_graph2

    metric_object = "links"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    defenses = [
        ("local", "neighbors", "links"),
        ("local", "jaccard", "links"),
        ("local", "cosine", "links"),
        # ("local", "ccpa", "links"),

        ("distributed", "neighbors", "links"),
        ("distributed", "cosine", "links"),
        ("distributed", "jaccard", "links"),
        # ("distributed", "adamic_adar", "links"),
        # ("distributed", "resource_allocation", "links"),
        # ("local", "svd", "links"),
    ]

    entered_thresholds = args.thresholds
    neighbors_links_thresholds = [i for i in range(5)]
    neighbors_features_thresholds = [i for i in range(5)]

    jaccard_links_thresholds = [i/100 for i in range(entered_thresholds[0], entered_thresholds[1], entered_thresholds[2])]
    # jaccard_links_thresholds = [i/100 for i in range(10)]
    jaccard_features_thresholds = [i/100 for i in range(entered_thresholds[0], entered_thresholds[1], entered_thresholds[2])]

    cosine_links_thresholds = [i/100 for i in range(entered_thresholds[0], entered_thresholds[1], entered_thresholds[2])]
    # cosine_links_thresholds = [i/100 for i in range(10)]
    cosine_features_thresholds = [i/100 for i in range(entered_thresholds[0], entered_thresholds[1], entered_thresholds[2])]

    adamic_adar_links_thresholds = [i/100 for i in range(entered_thresholds[0], entered_thresholds[1], entered_thresholds[2])]

    resource_allocation_links_thresholds = [i/100 for i in range(entered_thresholds[0], entered_thresholds[1], entered_thresholds[2])]

    svd_k = [i for i in range(5, 51, 5)]

    if dataset in ["acm", "cora", "citeseer", "pubmed", "polblogs", "flickr", "blogcatalog"]:
        data = Dataset(root="/tmp", name=dataset, seed=seed, setting="gcn")
        groundtruth_adj, features, labels = data.adj, data.features, data.labels

    if dataset == "computers":
        computers = AmazonPyg(root='/tmp', name='computers')
        data = Pyg2Dpr(computers)
        groundtruth_adj, features, labels = data.adj, data.features, data.labels

    if dataset == "syn":
        generator = SyntheticGraph(3000, 500)
        groundtruth_adj, features, labels = generator.generate()

    groundtruth_adj, features, labels = data.adj, data.features, data.labels
    
    idx_train_str = open(f"../data/{dataset}_data/training_nodes.txt").readlines()
    idx_train =  [eval(i) for i in idx_train_str]

    idx_test_str = open(f"../data/{dataset}_data/testing_nodes.txt").readlines()
    idx_test =  [eval(i) for i in idx_test_str]
    
    idx_val_str = open(f"../data/{dataset}_data/eval_nodes.txt").readlines()
    idx_val =  [eval(i) for i in idx_val_str]

    attacked_nodes_str = open(f"../data/{dataset}_data/attacked_nodes.txt").readlines()
    attacked_nodes =  [eval(i) for i in attacked_nodes_str]



    density = 1
    results_path = "../results/"+args.expe_name

    if not os.path.exists(results_path):
        os.makedirs(results_path)    
        
    with open(results_path + "/" + dataset + "-ptb_rate_graph1_" + str(perturbation_rate_graph1) + "-ptb_rate_graph2_" + str(perturbation_rate_graph2) + "-ppt_"+str(
                args.proportion) + "-sample_" + str(density) + "-metric_" + metric_object + ".csv",
              "a+") as logs_of_defenses_with_attacks:

        logs_of_defenses_with_attacks.write(
            "run,"
            "threshold,"
            "attack,"
            "setting,"
            "metric,"
            "object,"
            "alpha,"
            "accuracy_graph1,"
            "accuracy_graph2,"
            "attack_dur_graph1,"
            "attack_dur_graph2,"
            "defense_dur_graph1,"
            "defense_dur_graph2,"
            "train_dur_graph1,"
            "train_dur_graph2,"
            "TP_graph1,"
            "TP_graph2,"
            "FP_graph1,"
            "FP_graph2\n")

        for run in range(number_of_runs):
            graph1_adj, graph2_adj = split_dataset(groundtruth_adj, args.proportion)

            # sp.save_npz("graph1.npz", graph1_adj)
            # sp.save_npz("graph2.npz", graph2_adj)

            print("Number of nodes                           :", groundtruth_adj.shape[0])
            print("Number of links in ground truth :         :", len(groundtruth_adj.nonzero()[0]) / 2)
            print("Number of features in ground truth graph  :", features.shape[1])
            print("Number of links of graph1                 :", len(graph1_adj.nonzero()[0]) / 2)
            print("Number of links of graph2                 :", len(graph2_adj.nonzero()[0]) / 2)

            run_pipeline(attacks, defenses, perturbation_rate_graph1, perturbation_rate_graph2)

    duration_seconds = time.time() - exp_start
    hours, minutes, seconds = convert_duration(duration_seconds)
    print(f"Duration: {hours} hours, {minutes} minutes, {seconds} seconds")