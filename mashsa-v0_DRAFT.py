#print ('hello')
import networkx as nx
from deeprobust.graph.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse


data = Dataset(root='D:/Docs/UQAM\Memoire/cryptograph/cryptograph-utility/code', name="polblogs", setting="gcn")
groundtruth_adj, features, labels = data.adj, data.features, data.labels

#hypothes 1:
# find les y s qui ne sont pas voisin de target avec class X
# trouver les y qui sont le plus fort (have more voisin de y et moins de voisin de x)
# on donne les output de l'attaque sur le graphe en csv(ReadCSVtoMatrix) et on le passe a Cryptograph

#networkx   ------------------- install and test so powerful on graphs.

#find the non_voisin noeuds with label y for target_node 
def find_non_neighbor_class_y(graph, target_node):
    non_neighbor_class_y = []
    for node in graph.nodes():
        if node.label == 'y' and not graph.has_edge(target_node, node):
            non_neighbor_class_y.append(node)
    return non_neighbor_class_y


#find les y most fort = plus de voisins y et moins de voisin x: at the same time
# we pass non_neighbor_class_y as argument of this:

def find_nodes_with_max_y_and_min_x_neighbors(graph, non_neighbor_class_y, n ):
   # max_y_neighbors = -1
    #min_x_neighbors = float('inf')
    #nodes_with_max_y_neighbors = []
    #nodes_with_min_x_neighbors = []
    nodes_with_max_y_and_min_x_neighbors = []

    for node in non_neighbor_class_y.nodes():
        #for nodetest in graph.nodes():
            #if nodetest.label == 'y':
            num_y_neighbors = sum(1 for neighbor in graph.neighbors(node) if neighbor.label == 'y')
            num_x_neighbors = sum(1 for neighbor in graph.neighbors(node) if neighbor.label == 'x')

            p = num_y_neighbors/num_x_neighbors
            # if num of y neighbors are more than x neighbors we accept the node:
            if p > 1 :
                nodes_with_max_y_and_min_x_neighbors.append(node)
    
    nodes_with_max_y_and_min_x_neighbors= sorted(nodes_with_max_y_and_min_x_neighbors, key=lambda node: sum(1 for neighbor in graph.neighbors(node) if neighbor.label == 'y'), reverse=True)
    return nodes_with_max_y_and_min_x_neighbors[:n]

#attack V_0 Mahsa

#choisir budget first nodes from nodes_with_max_y_and_min_x_neighbors:

# def select_y (nodes_with_max_y_and_min_x_neighbors, budget) :
#     selected_y = find_nodes_with_max_y_and_min_x_neighbors(graph, non_neighbor_class_y, budget)
#     return selected_y

def insert_edge(graph, budget, target_node, node2):
    inserted_edges= 0
    for node in node2: 
        if inserted_edges > budget : 
            break
        graph.add_edge(target_node, node)
        inserted_edges += 1
    return graph



def convert(graph):
    # Convert the graph to a CSR matrix
    adjacency_matrix = nx.adjacency_matrix(graph)
    # Convert the adjacency matrix to CSR format
    adjacency_matrix_csr = csr_matrix(adjacency_matrix)
    return adjacency_matrix_csr





           #if num_y_neighbors > max_y_neighbors:
             #   max_y_neighbors = num_y_neighbors
           #     nodes_with_max_y_neighbors = [node]
           # elif num_y_neighbors == max_y_neighbors:
           #     nodes_with_max_y_neighbors.append(node)

          #  if num_x_neighbors < min_x_neighbors:
          #      min_x_neighbors = num_x_neighbors
          #      nodes_with_min_x_neighbors = [node]
           # elif num_x_neighbors == min_x_neighbors:
          #      nodes_with_min_x_neighbors.append(node)
            

# ba union ham mishe 
                
    # Filter nodes with maximum number of y neighbors to only keep those with the minimum number of x neighbors
    #nodes_with_max_y_and_min_x_neighbors = [node for node in  nodes_with_max_y_and_min_x_neighbors 
     #if sum(1 for neighbor in graph.neighbors(node) if neighbor.label == 'x') == min_x_neighbors]




# enlever les liens : trouver les noeuds x qui sont voisin de TX
     #trouver les noeuds x qui ont le plus voisin de x et moins de y , enlever les liens entre eux et TX 

def find_neighbor_class_x(graph, target_node):
    neighbor_class_x = []
    for node in graph.nodes():
        if node.label == 'x' and graph.has_edge(target_node, node):
            neighbor_class_x.append(node)
    return neighbor_class_x




def remove_edge(graph, budget, target_node, node2):
    removed_edges= 0
    for node in node2: 
        if removed_edges > budget : 
            break
        graph.remove_edge(target_node, node)
        removed_edges += 1
    return graph


#export graph to the csv: ????????? is it what cryptograph uses ??????

def export (graph, path):
    nx.write_edgelist(graph, path, delimiter=',', data='False')









# hypothese 2 : travailler sur les liens entre les autre noeuds non juste les noeuds target et autre : indetectable

# l'etape prochaine : contourner cryptoGraohe: choisir les liens et noeuds qui ont le plus voisin avec TX pour 
# pour evaluation : evaluer success 
     # Il faudrait évaluer le succès de l’adversaire en fonction du budget sur le nombre de liens qu’il peut modifier 
     # Un point important aussi serait de voir si l’adversaire ajoute tous les liens un par un (évaluer Crypto’Graph après chaque ajout de lien)
     # ou en une seule fois, ce qui pourrait aussi impacter les stratégies d’ajout de liens.