from tkinter import Frame
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import time
import community 
from itertools import product, combinations
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg





def louvain_algorithm(graph):

    graph = graph.to_undirected()
    partition = {node: node for node in graph.nodes()}

    best_partition = partition

    best_modularity = -1

    modularity_values = []

    while True:
        communities = community.best_partition(graph, partition=partition)

   
        modularityy = community.modularity(communities, graph)

        modularity_values.append(modularityy)

        if modularityy > best_modularity:
            best_partition = communities

            best_modularity = modularityy

        elif communities == best_partition:
            break

        partition = communities

    return best_partition,best_modularity


def compute_by_louvain(nodes_df,edges_df,directed=False , weighted=False ):

    if directed and weighted :
       G,adj_matrix = directed_weighted(nodes_df,edges_df)
    elif (not directed) and weighted :
       G,adj_matrix = undirected_weighted(nodes_df,edges_df)
    elif directed and (not weighted):
       G,adj_matrix = directed_unweighted(nodes_df,edges_df)
    else :
       G,adj_matrix = undirected_unweighted(nodes_df,edges_df)


    communities,modularity_value = louvain_algorithm(G)  
    unique_values = set(communities.values())   
    number_of_communities = len(unique_values)

    
    list_communities =[[] for i in range(len(unique_values))]
    for key,value in communities.items():
        list_communities[value].append(key)

    return number_of_communities,modularity_value,list_communities,G


def sort_by_second(elem):
    return elem[1]

def modularity_matrix(adj_matrix : np.ndarray) -> np.ndarray:
    k_i = np.expand_dims(adj_matrix.sum(axis=1), axis=1)
    k_j = k_i.T
    #norm = 1 / k_i.sum()
    norm = 1 / k_i.sum()
    K = norm * np.matmul(k_i, k_j)

    return norm * (adj_matrix - K)

def modularity(mod_matrix : np.ndarray, communities : list) -> float:
    C = np.zeros_like(mod_matrix)
    for community in communities:
        if isinstance(community, int):  
             community = [community]  
        for i, j in combinations(community, 2):
            C[i, j] = 1.0
            C[j, i] = 1.0

    return np.tril(np.multiply(mod_matrix, C), 0).sum()

def cut_edges(G):
    #get number of subgraphs 
    init_num_comps = nx.number_connected_components(G)
    curr_num_comps = init_num_comps
    cuts_count =0

    while curr_num_comps <= init_num_comps:
        bw_centralities = nx.edge_betweenness_centrality(G, weight="weight")
        
        #sort betweenness values in decreasing order 
        #bw_centralities.items => list of tuples contain 0 edges and 1 betweennes value
        bw_centralities = sorted(bw_centralities.items(),key=sort_by_second,reverse=True)
       # bw_centralities = sorted(
       #     bw_centralities.items(),
       #     key=lambda e: e[1],
       #     reverse=True
       #  )

        #remove the same bigger values in list 
        max_bw = None
        for edge, bw in bw_centralities:
            bw=round(bw,5)
            if max_bw is None:
                max_bw = bw

            if max_bw == bw:
                G.remove_edge(*edge)
                cuts_count+=1

            else:
                break

        curr_num_comps = nx.number_connected_components(G)
    print("number of cuts :" ,cuts_count)
    return G


def girvan_newman(adj_matrix : np.ndarray, n : int = None) -> list:
    M = modularity_matrix(adj_matrix)
    G = nx.Graph(adj_matrix)
    num_nodes = G.number_of_nodes()
    G.remove_edges_from(nx.selfloop_edges(G))

    best_P = list(nx.connected_components(G)) 
    best_Q = modularity(M, best_P)
    P_history = [best_P]
    Q_history = [best_Q]
    while True:
        last_P = P_history[-1]
        if not n and len(last_P) == num_nodes:
            return best_P,best_Q 
        elif n and len(last_P) >= n:
            return last_P,best_Q

        G = cut_edges(G)
        P = list(nx.connected_components(G))
        print("len now :",len(P))
        print("best q :",best_Q)
        Q = modularity(M, P)
        if Q >= best_Q:
            best_Q = Q
            best_P = P

        P_history.append(P)
        Q_history.append(Q)


def compute_by_girvenewman (nodes_df,edges_df,directed=False , weighted=False, n=None ):

    if directed and weighted :
       G,adj_matrix = directed_weighted(nodes_df,edges_df)
    elif (not directed) and weighted :
       G,adj_matrix = undirected_weighted(nodes_df,edges_df)
    elif directed and (not weighted):
       G,adj_matrix = directed_unweighted(nodes_df,edges_df)
    else :
       G,adj_matrix = undirected_unweighted(nodes_df,edges_df)

    communities , best_modularity = girvan_newman(adj_matrix,n)
    number_of_communities = len(communities)
    return number_of_communities,best_modularity,communities,G

    

def directed_weighted (nodes_df,edges_df): 

    node_to_index = {node_id: i for i, node_id in enumerate(nodes_df['ID'])}
    num_nodes = len(nodes_df)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i, edge in edges_df.iterrows():
        source_index = node_to_index[edge["Source"]]
        target_index = node_to_index[edge["Target"]]
        adj_matrix[source_index, target_index] += 1
        #adj_matrix[target_index, source_index] = 1  

    G = nx.DiGraph(adj_matrix) 
    return G ,adj_matrix


def directed_unweighted(nodes_df,edges_df):
    
    node_to_index = {node_id: i for i, node_id in enumerate(nodes_df['ID'])}
    num_nodes = len(nodes_df)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i, edge in edges_df.iterrows():
        source_index = node_to_index[edge["Source"]]
        target_index = node_to_index[edge["Target"]]
        adj_matrix[source_index, target_index] = 1
        #adj_matrix[target_index, source_index] = 1  

    G = nx.DiGraph(adj_matrix) 
    return G ,adj_matrix


def undirected_weighted (nodes_df,edges_df):

    node_to_index = {node_id: i for i, node_id in enumerate(nodes_df['ID'])}
    num_nodes = len(nodes_df)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i, edge in edges_df.iterrows():
        source_index = node_to_index[edge["Source"]]
        target_index = node_to_index[edge["Target"]]
        adj_matrix[source_index, target_index] += 1
        #adj_matrix[target_index, source_index] += 1  
    G = nx.Graph(adj_matrix) 
    return G ,adj_matrix


def undirected_unweighted (nodes_df,edges_df):

    node_to_index = {node_id: i for i, node_id in enumerate(nodes_df['ID'])}
    num_nodes = len(nodes_df)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i, edge in edges_df.iterrows():
        source_index = node_to_index[edge["Source"]]
        target_index = node_to_index[edge["Target"]]
        adj_matrix[source_index, target_index] = 1
        #adj_matrix[target_index, source_index] = 1  

    G = nx.Graph(adj_matrix) 
    return G ,adj_matrix


def visulize_community (window , communities,G):
    #plot graph inside the window
    frame = Frame(window)
    frame.pack()
    fig, ax = plt.subplots()


    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    pos = nx.spring_layout(G)
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=colors[i % len(colors)])
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    #plt.show()
        

    # Create a Matplotlib canvas and embed it in the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()    


""" 

nodes_df = pd.read_csv(r'/home/rengo/Downloads/social project/TestCase/UndirectedData/RomeoAndJuliet/nodes.csv')
edges_df = pd.read_csv(r'/home/rengo/Downloads/social project/TestCase/UndirectedData/RomeoAndJuliet/edges.csv')
 
  #number_of_comm,mod , comm =  compute_by_louvain(nodes_df,edges_df,directed=False,weighted=True)
number_of_comm,mod , comm,G =  compute_by_girvenewman(nodes_df,edges_df,directed=False,weighted=True,n=) 

print(f"number of communities : {number_of_comm}")
print(f"modularity : {mod}") 
 """