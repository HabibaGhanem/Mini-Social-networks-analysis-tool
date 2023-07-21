from tkinter import *
#from PIL import ImageTk ,Image 
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt 
import networkx as nx
import pandas as pd 
from tkinter import ttk
import igraph
import leidenalg  
import Community_detection as ts 
from cdlib import algorithms, evaluation,viz, NodeClustering

""""
def community_detection(algorithm):
    if(algorithm == "louvain"):
        communities = algorithms.louvain(G)

    elif(algorithm == "Class"):
        classes = nodes['Class'].unique()
        print(classes)
        comm = []
        for c in classes:
            selected_nodes = nodes[nodes['Class'] == c]
            ids = selected_nodes['ID'].tolist()
            print(c)
            print(ids)
            comm.append(ids)
        print(comm)
        communities = NodeClustering(comm,G)

    #viz.plot_network_clusters(G, communities, pos, plot_labels=True)
    #plt.show()
    #print(communities.communities)
    return communities
"""

root =Tk()
root.title("mini gephi")
root.geometry("1000x1000")


tabControl = ttk.Notebook(root)
nodes_tab = ttk.Frame(tabControl)
edges_tab = ttk.Frame(tabControl)
tabControl.add(nodes_tab, text='Nodes')
tabControl.add(edges_tab, text='Edges')
tabControl.pack()
tabControl.place(x=0,y=0,width=400,height=300)


type_of_graph = IntVar()
nodes_path = ""
edges_path= ""


def upload_file():

    new_window = Toplevel(root)
    new_window.title("upload files")
    new_window.geometry("500x500")

    def upload_Btn (button_id):
        #filename = filedialog.askopenfilename(initialdir="/home/rengo/Downloads", title="Select A File", filetypes=(("jpg files", ".jpg"),("all files", ".*")))
        #print(f" {button_id} : {filename}")
        global nodes_path
        global edges_path
        if(button_id == "nodes"):
            nodes_path= filedialog.askopenfilename(initialdir="C:\\Users\\hp\\Downloads\\social project")

        else:
            edges_path=filedialog.askopenfilename(initialdir="C:\\Users\\hp\\Downloads\\social project")

    def done ():
        new_window.destroy()

        print("nodes :",nodes_path)
        print("edges :",edges_path)
        print("type of graph :",type_of_graph.get())



        def selection_nodestab():
            #print(combobox_nodestab.get())
            frame = Frame(root)
            frame.pack()
            frame.place(x=400,y=100,width=500,height=500)
            fig, ax = plt.subplots()
            if(combobox_nodestab.get()=="Degree"):
                Adjusting_node_size_based_on_node_degree()
            elif(combobox_nodestab.get()=="In-Degree"):
                Adjusting_node_size_based_on_node_in_degree()
            elif(combobox_nodestab.get()=="Out-Degree"):
                Adjusting_node_size_based_on_node_out_degree()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

        def selection_edgestab():
            #print(combobox_edgestab.get())
            frame = Frame(root)
            frame.pack()
            frame.place(x=400, y=100, width=500, height=500)
            fig, ax = plt.subplots()
            if(combobox_edgestab.get()=="Weight"):
                Adjusting_edge_width_based_on_edge_weight()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

        def Adjusting_node_size_based_on_node_degree():
            nx.draw(G, pos, with_labels=True, node_size=[(degree + 1) * 50 for (node, degree) in nx.degree(G)])
            #plt.show()

        def Adjusting_node_size_based_on_node_in_degree():
            nx.draw(G,pos, with_labels=True, node_size=[(degree+1) * 50 for (node, degree) in G.in_degree()])
            #plt.show()

        def Adjusting_node_size_based_on_node_out_degree():
            nx.draw(G,pos, with_labels=True, node_size=[(degree+1) * 50 for (node, degree) in G.out_degree()])
            #plt.show()

        def Adjusting_edge_width_based_on_edge_weight():
            nx.draw(G,pos, node_size=50, with_labels=True, width=[edge[2] for edge in G.edges(data='Weight')])
            #plt.show()

        

        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)

        if 'Weight' in edges:
            edgesWithWeights = edges

        else:
            edgesWithWeights = edges.groupby(['Source', 'Target']).size().reset_index(name='Weight')
            edgesWithWeights.to_csv(index=False)

        if not type_of_graph.get():
            G = nx.from_pandas_edgelist(edgesWithWeights, 'Source', 'Target','Weight', create_using=nx.Graph())
            combobox_nodestab = ttk.Combobox(nodes_tab,values=["Degree"])
        else:
            G = nx.from_pandas_edgelist(edgesWithWeights, 'Source', 'Target','Weight', create_using=nx.DiGraph())
            combobox_nodestab = ttk.Combobox(nodes_tab, values=["Degree","In-Degree","Out-Degree"])

        #print(len(list(G.nodes)))
        G.add_nodes_from(nodes['ID'].tolist())
        #print(len(list(G.nodes)))
        pos = nx.spring_layout(G)
        frame = Frame(root)
        frame.pack()
        frame.place(x=400, y=100, width=500, height=500)
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True,node_size=50)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        ttk.Label(nodes_tab,text="Adjusting node size based on:").place(x=20,y=60)
        combobox_nodestab.place(x=75,y=100)
        button_nodestab = ttk.Button(nodes_tab,text="Apply",command=selection_nodestab)
        button_nodestab.place(x=100,y=140)

        ttk.Label(edges_tab,text="Adjusting edge thickness based on:").place(x=20,y=60)
        combobox_edgestab = ttk.Combobox(edges_tab,values=["Weight"])
        combobox_edgestab.place(x=75,y=100)
        button_edgestab = ttk.Button(edges_tab,text="Apply",command=selection_edgestab)
        button_edgestab.place(x=100,y=140)

    def when_change(value):
        print(value)

    Button(new_window,text="upload nodes",name="nodes",command=lambda: upload_Btn("nodes")).pack()
    Button(new_window,text="upload edges",name="edges",command=lambda: upload_Btn("edges")).pack()
    

    Label(new_window, text="directed or undirected:").pack()


    directed_button = Radiobutton(new_window, text="Directed", variable=type_of_graph, value=1,command=lambda: when_change(type_of_graph.get()))
    directed_button.pack()

    undirectd_button = Radiobutton(new_window, text="Undirected", variable=type_of_graph, value=0,command=lambda: when_change(type_of_graph.get()))
    undirectd_button.pack()

    Button(new_window,text="Done",command=done).pack()

def girven_newman():
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    




    girven_window = Toplevel(root)
    girven_window.title("girven result")
    girven_window.geometry("800x800")

    e = Entry(girven_window,width=50)
    e.insert(0,"how many cluster you need")
    e.pack()
    def run_girven():
        n=None
        print(e.get())
        if e.get().isspace() or len(e.get())==0:
            n=None 
        else:
            n=int(e.get())
        
        number_of_communities , modularity , communities,G = ts.compute_by_girvenewman(nodes_df,edges_df,type_of_graph.get(),True,n=n)


        #plot graph inside the window
        frame = Frame(girven_window)
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
        
        Label(girven_window, text=f"modularity value : {modularity}").pack()
        Label(girven_window, text=f"number of communities : {number_of_communities}").pack()

        communities_NodeCluster = NodeClustering(communities,G)
        evaluationn (girven_window,communities_NodeCluster,G)




    entry_button = Button(girven_window,text="run girven newman",command=run_girven).pack()

def louvain():
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    number_of_communities , modularity , communities,G =    ts.compute_by_louvain(nodes_df,edges_df,type_of_graph.get(),True)
    print(f"number_of_communities: {number_of_communities}")
    print(f"modularity : {modularity}")



    louvain_window = Toplevel(root)
    louvain_window.title("louvain result")
    louvain_window.geometry("800x800")

    #plot graph inside the window
    frame = Frame(louvain_window)
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
    
    Label(louvain_window, text=f"modularity value : {modularity}").pack()
    Label(louvain_window, text=f"number of communities : {number_of_communities}").pack()

    communities_NodeCluster = NodeClustering(communities,G)


    evaluationn (louvain_window,communities_NodeCluster,G) 
    
    """ evaluation_type = ttk.Combobox(louvain_window,values=["nmi","internal edge denisty","avgDist","modularity"])
    evaluation_type.pack()

    eval_st =StringVar()
    
    def community_detection_evaluation():
        print("suiii")
        


    b=Button(louvain_window,text="calc",command=lambda: community_detection_evaluation())
    b.pack() """

def evaluationn (window,communities,G) :

    
    evaluation_type = ttk.Combobox(window,values=["nmi","internal edge denisty","avgDist"])
    evaluation_type.pack()

    
    def community_detection_evaluation(ev,communities,G):
        ##modularity (Internal evaluation)
        eval_st = " "
       

        ##NMI (external evaluation)
        if(ev == "nmi"):
            #first method => comparing different graph partition to assess their resemblance
            leiden_communities = algorithms.leiden(G)
            #print(len(leiden_communities.communities))
            NMI = evaluation.normalized_mutual_information(communities,leiden_communities)
            #print(NMI)
            #second method applying nmi by using a class attribute
            #class_communities = community_detection("Class")
            #NMI = evaluation.normalized_mutual_information(communities, class_communities)
            #print(NMI)
            eval = NMI
            eval_st = str(eval)

        ##Internal edge distance (Internal evaluation)
        elif(ev == "internal edge denisty"):
            internal_edge_denisty = evaluation.internal_edge_density(G,communities,summary=False)
            eval = internal_edge_denisty
            #print(internal_edge_denisty)
            eval_st = str(eval)


        ##Average distance (Internal evaluation)
        #The average distance of a community is defined average path length across all possible pair of nodes composing it
        elif(ev == "avgDist"):
            avg_dist = evaluation.avg_distance(G,communities,summary=False)
            eval = avg_dist
            #print(avg_dist)
            eval_st = str(eval)

        
        Label(window,text= f"{ev} : {eval_st} ").pack()
        

    #lambda: when_change(type_of_graph.get())
    b=ttk.Button(window , text="calc",command=lambda:community_detection_evaluation(evaluation_type.get(),communities,G) ).pack()

def closeness_centrality():
    window = tk.Tk()
    window.title("Closeness Centrality")
    window.geometry('600x600')

    # add a label
    label = tk.Label(window, text='Filter by closeness centrality:')
    label.pack()

    # add an entry box
    entry_box = tk.Entry(window)
    entry_box.pack()

    # add a button
    def filter_nodes():
        centrality_threshold = float(entry_box.get())
        filtered_nodes = [node for node, centrality in nx.closeness_centrality(graph).items() if
                          centrality > centrality_threshold]
        display(filtered_nodes)
        plot_graph(filtered_nodes)

    button = tk.Button(window, text='Filter', command=filter_nodes)
    button.pack()

    # add a canvas
    canvas = tk.Canvas(window)
    canvas.pack()

    # generate the graph
    df = pd.read_csv(edges_path)
    if type_of_graph.get() : 
        print("ungraphhhhhhhhhhhhhhhhh")
        graph = nx.DiGraph()
    else:
        print("graphhhhhhhhhhhhhhhhh")
        graph = nx.Graph()    
    
    for i, row in df.iterrows():

        graph.add_edge(row['Source'], row['Target'])

    nx.draw(graph, with_labels=True)

    def plot_graph(filtered_nodes):
        window1 = tk.Tk()
        window1.title("Closeness Centrality Graph")
        window1.geometry('700x700')
        frame = Frame(window1)
        frame.pack()
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue')
        nx.draw_networkx_nodes(graph, pos, node_color='red', nodelist=filtered_nodes)
        nx.draw_networkx_edges(graph, pos, alpha=0.5)
        nx.draw_networkx_labels(graph, pos)

        # Create a Matplotlib canvas and embed it in the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        window1.mainloop()

    def display(filtered_nodes):
        listbox = tk.Listbox(window)
        for item in filtered_nodes:
            listbox.insert(tk.END, item)
        listbox.pack()



    # start the tkinter main loop
    window.mainloop()

def betweenness_centrality():
    window = tk.Tk()
    window.title("betweenness Centrality")
    window.geometry('600x600')

    # add a label
    label = tk.Label(window, text='Filter by betweenness centrality:')
    label.pack()

    # add an entry box
    entry_box = tk.Entry(window)
    entry_box.pack()

    # add a button
    def filter_nodes():
        centrality_threshold = float(entry_box.get())
        filtered_nodes = [node for node, centrality in nx.betweenness_centrality(graph,normalized=False).items() if
                          centrality > centrality_threshold]
        display(filtered_nodes)
        plot_graph(filtered_nodes)

    button = tk.Button(window, text='Filter', command=filter_nodes)
    button.pack()

    # add a canvas
    canvas = tk.Canvas(window)
    canvas.pack()

    # generate the graph
    df = pd.read_csv(edges_path)
    if type_of_graph.get() : 
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()    
    
    for i, row in df.iterrows():
        graph.add_edge(row['Source'], row['Target'])
    nx.draw(graph, with_labels=True)

    def plot_graph(filtered_nodes):
        window1 = tk.Tk()
        window1.title("Betweenness Centrality Graph")
        window1.geometry('700x700')
        frame = Frame(window1)
        frame.pack()
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue')
        nx.draw_networkx_nodes(graph, pos, node_color='red', nodelist=filtered_nodes)
        nx.draw_networkx_edges(graph, pos, alpha=0.5)
        nx.draw_networkx_labels(graph, pos)

        # Create a Matplotlib canvas and embed it in the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        window1.mainloop()

    def display(filtered_nodes):
        listbox = tk.Listbox(window)
        for item in filtered_nodes:
            listbox.insert(tk.END, item)
        listbox.pack()

    # start the tkinter main loop
    window.mainloop()

def degree_centrality():
    window = tk.Tk()
    window.title("degree Centrality")
    window.geometry('600x600')

    # add a label
    label = tk.Label(window, text='Filter by degree centrality:')
    label.pack()

    # add an entry box
    entry_box = tk.Entry(window)
    entry_box.pack()

    # add a button
    def filter_nodes():
        centrality_threshold = float(entry_box.get())
        if type_of_graph.get():
            filtered_nodes = [node for node, centrality in nx.in_degree_centrality(graph).items() if
                              centrality > centrality_threshold]
            display(filtered_nodes)
            plot_graph(filtered_nodes)
        else:
            filtered_nodes = [node for node, centrality in nx.degree_centrality(graph).items() if
                              centrality > centrality_threshold]
            display(filtered_nodes)
            plot_graph(filtered_nodes)

    button = tk.Button(window, text='Filter', command=filter_nodes)
    button.pack()

    # add a canvas
    canvas = tk.Canvas(window)
    canvas.pack()

    # generate the graph
    df = pd.read_csv(edges_path)
    if type_of_graph.get():
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    for i, row in df.iterrows():
        graph.add_edge(row['Source'], row['Target'])
    nx.draw(graph, with_labels=True)

    def plot_graph(filtered_nodes):
        window1 = tk.Tk()
        window1.title("Degree Centrality Graph")
        window1.geometry('700x700')
        frame = Frame(window1)
        frame.pack()
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue')
        nx.draw_networkx_nodes(graph, pos, node_color='red', nodelist=filtered_nodes)
        nx.draw_networkx_edges(graph, pos, alpha=0.5)
        nx.draw_networkx_labels(graph, pos)

        # Create a Matplotlib canvas and embed it in the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        window1.mainloop()

    def display(filtered_nodes):
        listbox = tk.Listbox(window)
        for item in filtered_nodes:
            listbox.insert(tk.END, item)
        listbox.pack()

    # start the tkinter main loop
    window.mainloop()

def PageRank():
    window = tk.Tk()
    window.title("Page Rank")
    window.geometry('600x600')

    # add a label
    label = tk.Label(window, text='Highest Page Rank')
    label.pack()

    # add a button
    def filter_nodes():
        page_rank=nx.pagerank(graph)
        max_pr = max(page_rank.values())
        filtered_nodes = [node for node, pagerank in nx.pagerank(graph).items() if
                          pagerank==max_pr]
        display(filtered_nodes)
        plot_graph(filtered_nodes)

    button = tk.Button(window, text='Display', command=filter_nodes)

    button.pack()
    # add a canvas
    canvas = tk.Canvas(window)
    canvas.pack()

    # generate the graph
    df = pd.read_csv(edges_path)
    if type_of_graph.get() : 
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()    
    
    for i, row in df.iterrows():
        graph.add_edge(row['Source'], row['Target'])
    nx.draw(graph, with_labels=True)

    def plot_graph(filtered_nodes):
        window1 = tk.Tk()
        window1.title("Page Rank Graph")
        window1.geometry('700x700')
        frame = Frame(window1)
        frame.pack()
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue')
        nx.draw_networkx_nodes(graph, pos, node_color='red', nodelist=filtered_nodes)
        nx.draw_networkx_edges(graph, pos, alpha=0.5)
        nx.draw_networkx_labels(graph, pos)

        # Create a Matplotlib canvas and embed it in the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        window1.mainloop()


    def display(filtered_nodes):

        messagebox.showinfo("Node", filtered_nodes)

#menu
my_menu =Menu(root)
root.config(menu=my_menu)

file_menu = Menu(my_menu)
my_menu.add_cascade(label="File",menu=file_menu)
file_menu.add_command(label="upload files",command=upload_file)
file_menu.add_command(label="Exit",command=root.quit)



community_detection = Menu(my_menu)
my_menu.add_cascade(label="community detection ",menu=community_detection)
community_detection.add_command(label="girven newman",command=girven_newman)
community_detection.add_command(label="louvain ",command=louvain)


measure_centrality = Menu(my_menu)
my_menu.add_cascade(label="measure centrality", menu=measure_centrality)
measure_centrality.add_command(label="degree centrality", command=degree_centrality)
measure_centrality.add_command(label="betweenness centrality", command=betweenness_centrality)
measure_centrality.add_command(label="closeness centrality ", command=closeness_centrality)

Page_Rank = Menu(my_menu)
my_menu.add_cascade(label="Page Rank", menu=Page_Rank)
Page_Rank.add_command(label="Page Rank", command=PageRank)




root.mainloop()