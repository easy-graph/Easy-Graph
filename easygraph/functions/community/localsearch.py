# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:00:36 2021
Updated on Sun Jun 09 12:33:06 2024

Local Search (LS) algorithm proposed in 
Dingyi Shi, Fan Shang, Bingsheng Chen, Paul Expert, Linyuan Lv, H. Eugene Stanley, Renaud Lambiotte, Tim S. Evans, Ruiqi Li, 
Local dominance unveils clusters in networks, Communications Physics, 2024, 7:170 [PDF: https://rdcu.be/dJxY0]

"Hidden directionality unifies community detection and cluster analysis"

@authors: Fan Shang & Tim S. Evans & Ruiqi Li & Dingyi Shi
"""

import os
import random
import easygraph as eg
import numpy as np
from queue import Queue
from datetime import datetime
# from LS_other_function import plot_combination

font = {'family': 'Times New Roman',
        'style': 'italic',
        'weight': 'normal',
        'size': 22,
        }

def plot_combination(
        x,
        y,
        text,
        x1,
        y1,
        text1,
        center_id,
        subplot_location,
        xlim_start_end,
        ylim_start_end,
        font_location,
        filepath='./',
        dataname='LS_default',
        save=False,
        show=False):
    '''
    input:
        x：节点的度值(数据类型：list)k
        y：节点的最短路径(数据类型：list)l
        x1：节点按照乘积~{k_i} * ~{l_i}的rank排序 (数据类型：list)
        y1：~{k_i} * ~{l_i}(数据类型：list)
        text：节点的id(数据类型：list)
        filepath：需要存储的文件路径(数据类型：str)
        center_id: LS算法识别的社团中心节点集合(数据类型：list)
        dataname: 当前网络的名称(数据类型：str)
        save：是否需要存储文件(数据类型：boolean)
    return:
       plot
    '''

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        import matplotlib.colors as mc
        import colorsys
    except ImportError as exc:
        raise ImportError("plot_combination requires matplotlib to be installed") from exc

    def adjust_lightness(color, amount=0.5):
        try:
            c = mc.cnames[color]
        except KeyError:
            c = color
        hls_color = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(hls_color[0], max(0, min(1, amount * hls_color[1])), hls_color[2])

    fig = plt.figure(figsize=(8, 7))
    basecolor = '#FFA900'
    edgecolor = adjust_lightness(basecolor, amount=1)
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    for i in range(len(x)):
        # ax.scatter(x[i], y[i], c=basecolor, marker='o', s=200, edgecolor=edgecolor)
        if text[i] in center_id:
            ax.text(x[i], y[i] + font_location, str(text[i]), ha='center', fontsize=12, fontweight='bold')
    ax.scatter(x, y, c=basecolor, marker='o', s=200)

    if np.max(np.array(x)) // 10 < 1:
        x_unit = 1
    else:
        x_unit = np.max(np.array(x)) // 10
    if np.max(np.array(y)) // 10 < 1:
        y_unit = 1
    else:
        y_unit = np.max(np.array(y)) // 10
    x_major_locator = MultipleLocator(x_unit)
    y_major_locator = MultipleLocator(y_unit)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xlim(xlim_start_end[0], max(x) + xlim_start_end[1])
    ax.set_ylim(ylim_start_end[0], max(y) + ylim_start_end[1])
    ax.set_xlabel(r'$k_i$', font)
    ax.set_ylabel(r'$l_i$', font)
    ax.tick_params(labelsize=16)

    font1 = {'family': 'Times New Roman',
             'style': 'italic',
             'weight': 'normal',
             'size': 16,
             }
    basecolor = '#A73489'
    edgecolor = adjust_lightness(basecolor, amount=1)
    #     left, bottom, width, height = 0.25,0.595,0.35,0.3 # darkar
    #     left, bottom, width, height = 0.18,0.55,0.35,0.3 # Abidjan
    #     left, bottom, width, height = 0.25,0.55,0.35,0.3 # Beijing
    # 添加子图
    left, bottom, width, height = subplot_location[0], subplot_location[1], subplot_location[2], subplot_location[3]
    ax1 = fig.add_axes([left, bottom, width, height])
    # 对x1,y1进行log-log处理
    x1_new = np.log(np.array(x1) + 1)
    y1_new = []
    y1_min = min(filter(lambda x: x > 0, y1))
    for i in range(len(y1)):
        if y1[i] != 0:
            y1_new.append(np.log(y1[i]))
        else:
            y1_new.append(np.log(y1_min / np.e))
    # for i in range(len(x1_new)):
    #     #     if text1[i] in center_id:
    #     #         ax1.scatter(x1_new[i], y1_new[i], color=basecolor, marker='^', s=20, edgecolor=edgecolor)
    #     #     else:
    #     #         ax1.scatter(x1_new[i], y1_new[i], color=basecolor, marker='o', s=2, edgecolor=edgecolor)
    #     # #             ax1.text(x1_new[i], y1_new[i]-0.1, str(int(text1[i])), ha='center', fontsize=10,fontweight='bold')
    ax1.scatter(x1_new, y1_new, color=basecolor, marker='o', s=2)
    center_x = []
    center_y = []
    for i in range(len(x1_new)):
        if text1[i] in center_id:
            center_x.append(x1_new[i])
            center_y.append(y1_new[i])
    ax1.scatter(center_x, center_y, color=basecolor, marker='^', s=20)

    ax1.set_xlabel(r'$\ln \, rank$', font1)
    ax1.set_ylabel(r'$\ln \, ( \~{k_i} \times \~{l_i} ) $', font1)
    ax1.tick_params(labelsize=16)
    fig.tight_layout()
    if save:
        os.makedirs(filepath, exist_ok=True)
        filename = os.path.join(filepath, f"{dataname}.pdf")
        fig.savefig(filename, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def max_degree_hierarchy_dag(G, selfloop_nodes=None):
    '''
    Create a maximum degree hierarchy DAG from a graph G

    All edges present are from a source node to neighbours which have a larger degree
    and the degree of these neigthbours is is larger than or equal to than the degree
    of all the neighbours of the source vertex.

    The difference from the full_degree_hierarchy_dag method is that this
    does not inlcude links to neighbours which have a higher degree than the source node
    but still that neighbouir has a degree which is less than the largest degree
    of all the neighbours.

	This subroutine create the DAG in Fig.1b in the maintext of our paper

    Input
    -----
    G -- a simple graph of one component

    Return
    ------
    D -- A directed acyclic graph
    '''
    D = eg.DiGraph()
    D.add_nodes_from(G)
    for v in G.nodes:
        # degree_list = [G.degree(nn) for nn in G.neighbors(v)]
        degree_list = []
        for nn in G.neighbors(v):
            if nn in selfloop_nodes:
                degree_list.append(G.degree()(nn) + 1)
            else:
                degree_list.append(G.degree()[nn])
        if len(degree_list) > 0:
            knnmax = max(degree_list)
            # print(G.degree()[v])
            if knnmax >= G.degree()[v]:  # can also use np.argmax() here
                # has neighbours with the largest degree so add all of the edges to this neighbour
                # here edge points from low degree to high degree, points towards tree root
                e_list = [(v, nn) for nn in G.neighbors(v) if G.degree()[nn] == knnmax and (not D.has_edge(nn, v))]
                D.add_edges_from(e_list)
        else:
            continue
    # print("! With "+str(G.number_of_nodes())+" nodes, from "+str(G.number_of_edges())+" to "+str(D.number_of_edges())+" edges in Maximum Degree DAG")
    #     print(D.edges())
    return D


# def full_degree_hierarchy_dag(G,selfloop_nodes=None):
# '''
# [DEPRECATED] Create a full degree hierarchy DAG from a simple graph G,
# which is a variant of the algorithm presented in our paper.
# All edges are directed from lower to higher degree nodes, only edges not included are those between equal degree nodes.
# Input: G -- a simple graph of one component
# Return: D -- A directed acyclic graph
# '''
# D = eg.DiGraph()
# D.add_nodes_from(G)
# for v in G.nodes:
# kv = G.degree(v)
# if v in selfloop_nodes:
# kv+=1
# e_list = [(v,nn) for nn in G.neighbors(v) if G.degree(nn)>kv]  # point to larger degree node
# D.add_edges_from( e_list )
# # print("! With "+str(G.number_of_nodes())+" nodes, from "+str(G.number_of_edges())+" to "+str(D.number_of_edges())+" edges in Full Degree DAG")
# return D

def degree_hierarchy_random_tree(G, maximum_tree=True, random_seed=None, selfloop_nodes=None):
    '''
    Create a degree hierarchy tree from a graph G.

    Unless seed=None, this uses a certain random number series to break ties
    where neighbours have same (maximum) degree and they are
    both at the same distance from a root node.

	This subroutine create the DAG comprising all short-dahsed-arrows in Fig.1c in the maintext of our paper

    Input
    -----
    G -- an simple graph of one component
    maximum_tree=True -- If true uses maximum dgree DAG as input, otherwise uses full degree DAG
    random_seed -- an specific integer to determine the random number series
    selfloop_nodes -- In the default setting (None), self-loops are not considered; if not None, self-loop will add influence (degree) to the node

    Return
    ------
    D, tree_edge_list

    D --- A directed acyclic graph (DAG)
    tree_edge_list --- list of edges in terms of node ID used in G of a shortest path tree in G
    '''
    if random_seed != None:
        random.seed(random_seed)

    if maximum_tree:
        D = max_degree_hierarchy_dag(G, selfloop_nodes)
        # D is a DAG in Fig. 1b in the main text of our paper
    # else:
    #     D=full_degree_hierarchy_dag(G,selfloop_nodes)
    # This is a DEPRECATED variant DAG (not the one we used in our paper)

    node_queue = Queue(maxsize=0)
    # start queue for BFS from all the root nodes
    # Each entry in queue is tuple (parent_node, node, shortest_distance_to_root)
    parent_node = None
    shortest_distance_to_root = 0
    for root_node in D:
        if D.out_degree()[root_node] == 0:
            # print("Adding root node "+str(root_node))
            node_queue.put((None, root_node, shortest_distance_to_root))
    number_of_ties = 0
    # now we have all local leaders in the queue

    while not node_queue.empty():
        parent_node, next_node, shortest_distance_to_root = node_queue.get()

        if "distancetoroot" in D.nodes[next_node]:
            if D.nodes[next_node]["distancetoroot"] < shortest_distance_to_root:
                continue  # already found a quicker way from next_node to a root node
            if D.nodes[next_node]["distancetoroot"] == shortest_distance_to_root:
                number_of_ties += 1
                if random.random() < 0.5:
                    continue  # a simple way to implement randomness where there is a choice of shortest path roots

        if parent_node == None:  # Must be a root node (i.e., a local leader)
            D.nodes[next_node]["rootnode"] = next_node
        else:
            D.nodes[next_node]["rootnode"] = D.nodes[parent_node]["rootnode"]
        D.nodes[next_node]["parentnode"] = parent_node
        D.nodes[next_node]["distancetoroot"] = shortest_distance_to_root
        # print(next_node,parent_node,shortest_distance_to_root)
        nn_list = [(next_node, nn, shortest_distance_to_root + 1) for nn in
                   D.predecessors(next_node)]  # get all neighbors of the next_node
        for nn in nn_list:
            node_queue.put(nn)
    tree_edge_list = []

    for node in D:
        parent_node = D.nodes[node]["parentnode"]
        if parent_node != None:
            tree_edge_list.append((parent_node, node))

    # print("! In degree_hierarchy_random_tree broke "+str(number_of_ties)+" ties at random")
    return D, tree_edge_list


# now we break all ties in Fig.1b (e.g., d->c,d->e; l->b,l->m), and tree_edge_list are short-dahsed-arrows in Fig.1c, and add information (rootnode,parentnode,distoroot), which are useful for community label backpropagation, of nodes in the DAG


# prelimenary functions for computing normalized ki*li (see Supplementary Information)
def get_indicator_rank(x):
    set_x = set(x)
    sorted_x = sorted(set_x, reverse=False)
    set_x_dict = {}
    k = 1
    for i in sorted_x:
        if i not in set_x_dict.keys():
            set_x_dict[i] = k
            k += 1
    rank_x = []
    for i in x:
        rank_x.append(set_x_dict[i])
    return rank_x


def get_square(x):
    square_x = []
    square_x = [np.power(i, 2) for i in x]
    return square_x


# min-max normalization
def standard_data(x):
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max - x_min == 0:
        trans_data = np.array([1 / len(x) for i in range(len(x))])
    else:
        trans_data = (x - x_min) / (x_max - x_min)
    return trans_data


# When there are multi-scale community structure in the network, we may want to get the first-level partion automatically sometimes. Here, we present a very simple algorithm to determine the number of first-level comunity centers: we calculate the differences between consecutive candicates in the decision graph (see Fig. 1f in our paper), and if the gap below a certain candicate is larger than the mean+std, then this gap might be a notable gap (this works relatively well for real networks we tested in our paper) #在存在多尺度社团(Multi-scale community structure)的情况下，根据y之间的差值自动选择第一层级的聚类中心的个数
# You can REPLACE this algorithm by a more rigorous and sophisticaed one, if you want to do automatic multi-scale community detection
# Otherwise, in our default setting, we will give the community partion at the finest resolution
def choose_center(multi_sort):
    y = multi_sort[:, 1]
    delta = []
    for i in range(len(y))[1:]:
        delta.append(abs(y[i] - y[i - 1]))
    # delta = np.array(delta) #
    delta_nozero = [i for i in delta if i != 0]
    delta_std = np.std(delta_nozero)
    center_num = 0
    for i in range(len(delta)):
        if delta[i] > delta_std + np.mean(delta_nozero):
            center_num = i + 1
            break
    return center_num


# Local-BFS (LBFS) from a local leader to determine its superior along hierarchy (or termed as finding hidden directionality of a local leader)
# This LBFS will stop right after enountering another local leader with a higher influence (e.g., influence can be measured by degree or other centrality measurements. This LBFS will not traverse the whole network, thus much less costly than normal BFS
def BFS_from_s(G, s, roots):
    '''
    input:
        G: graph  #图结构
        s: index of the source/start local leader (type:int) #[BFS开始的起始节点(数据类型：int)]
        roots: the set of all local leaders (type: list)
    return:
       w: the index of the superior local leader along the hierarchy; if no such superior, return itself #指向节点的id(数据类型：int),不存在时返回自己
       p: the shortest path from the local leader s to its superior local leader; when no superior, return -1 #最短路经长度(数据类型：int)，不存在时返回-1
    '''
    queue = []
    queue.append(s)
    seen = set()  # visited nodes in BFS #看是否访问过该结点
    seen.add(s)
    path_dict = {}  # path length to other nodes  #记录root到每个节点的距离
    path_dict[s] = 0
    while (len(queue) > 0):
        vertex = queue.pop(0)  # 保存第一结点，并弹出，方便把他下面的子节点接入
        neighbors = [(neighbor, G.degree()[neighbor]) for neighbor in list(G.adj[vertex]) if
                     neighbor not in seen]  # 子节点的数组
        nodes = [node[0] for node in
                 sorted(neighbors, key=lambda k: k[1], reverse=True)]  # the sorting here is not necessary
        #         print('nodes',vertex,nodes)
        for w in nodes:
            if w not in seen:  # not uncessary, just to make sure w is not in seen #判断是否访问过，使用一个数组
                path_dict[w] = path_dict[vertex] + 1
                queue.append(w)
                seen.add(w)
            if w in roots and G.degree()[w] > G.degree()[s]:  ###
                return w, path_dict[w]
    return s, -1


def hierarchical_degree_communities(
        G,
        center_num=None,
        auto_choose_centers=False,
        maximum_tree=True,
        isdraw=False,
        seed=None,
        self_loop=False,
        plot_filepath="./",
        plot_dataname="LS_default",
        plot_show=False,
):
    '''
    Produces hierarchical degree forest (HDF) of trees and hence communities.
	The main part of our Local Search (LS) algorithm

    Input
    -----
    G -- simple graph for which communities are required
    maximum_tree=True -- If true uses maximum dgree DAG as input, otherwise uses full degree DAG
    seed=None -- an integer to use as a seed to break ties at random.  Use None to remove random element
    self_loop -- If true means the self-loop makes sense
    plot_filepath -- directory to save the decision graph when isdraw is True
    plot_dataname -- filename (without extension) for the saved decision graph; saved as "<dataname>.pdf"
    plot_show -- whether to display the decision graph window when isdraw is True

    Output
    ------
    On screen statistics of communities

    '''
    # Ensure we work on an EasyGraph Graph copy so downstream methods (e.g., remove_edges_from) exist
    if not hasattr(G, "remove_edges_from"):
        converted = eg.Graph()
        try:
            converted.add_nodes_from(G.nodes)
        except Exception:
            converted.add_nodes_from(G.nodes())
        try:
            converted.add_edges_from(G.edges)
        except Exception:
            converted.add_edges_from(G.edges())
        G = converted
    else:
        G = G.copy()

    # Empty graph
    if not G.nodes:
        print("Warning: Empty graph detected. Returning empty results.")
        D = None
        center_dcd = set()
        y_dcd = set()
        y_partition = []
        grouped_dict = {}
        plot_combination_data = None
        return D, center_dcd, y_dcd, y_partition, grouped_dict, plot_combination_data

    # Disconnected graph
    if not G.edges:
        print("Warning: Disconnected graph detected.")
        D = None
        center_dcd = set(G.nodes.keys())
        y_dcd = set()
        y_partition = []
        grouped_dict = G.nodes
        plot_combination_data = None
        return D, center_dcd, y_dcd, y_partition, grouped_dict, plot_combination_data


    selfloop_edges = []
    if eg.number_of_selfloops(G) > 0:
        selfloop_edges = list(eg.selfloop_edges(G))
        G.remove_edges_from(selfloop_edges)
    selfloop_nodes = []
    for item in selfloop_edges:
        selfloop_nodes.append(item[0])
    if self_loop == False:
        selfloop_nodes = []

    start_time = datetime.now()
    treename = "Hierarchical Maximum Degree Forest"
    # treeabv="HMDF"
    if not maximum_tree:
        treename = "Hierarchical Full Degree Forest"
        # treeabv="HFDF"

    # print ("\n===== "+treename+" seed "+str(seed)+" =====")
    print("\n====Local Search Algorithm (random seed " + str(seed) + ")==========")
    print("Network: " + str(len(G.nodes)) + " nodes," + str(len(G.edges)) + " edges")
    D, tree_edge_list = degree_hierarchy_random_tree(G, maximum_tree=maximum_tree, random_seed=seed,
                                                     selfloop_nodes=selfloop_nodes)
    # D is the DAG comprising short-dahsed-arrows in Fig.1c in the main text of our paper
    # print("With "+str(G.number_of_nodes())+" nodes, now left with "+str(len(tree_edge_list))+" edges in tree" )

    # Now find all the nodes with the same root_node (i.e., local leaders)
    root_to_node = {}
    for node in D:
        if "rootnode" in D.nodes[node]:
            root_node = D.nodes[node]["rootnode"]
        else:
            print("*** ERROR Node " + str(node) + " has no rootnode")
            continue
        if root_node not in root_to_node:
            root_to_node[root_node] = []
        root_to_node[root_node].append(node)
    ##

    # determine centers from root_to_node
    # (1). using Local-BFS to determine the hidden directionalilty of each local leader (i.e., finding its superior among local leaders along the hierarchy & calculate shortest path lengh between it and its superior l_i  #通过local-BFS计算local leader的指向和最短路径
    root_to_node = {key: value for key, value in root_to_node.items() if len(value) > 1}
    Potential_Center = list(root_to_node.keys())
    # print("! Number of Communities (root nodes) found "+str(len(root_to_node)))
    # print("  Root Nodes: ",Potential_Center)

    root_number = len(root_to_node)
    root_decision = {}
    avg_l = 0
    # print('Intermediate process of determining the center: ')
    for node in root_to_node.keys():
        e, p = BFS_from_s(G, node, Potential_Center)  # Local-BFS, e is the superior, p is the path length to it
        root_decision[node] = [e, p, G.degree()[node]]

    # For local leaders with the maximal degree in the network and noisy nodes (isolated ones), setting their l_i as the maximum of l_i of all other local leaders [or the diameter of the network #度值最大的节点和噪声节点的最短路径长度设置为所有节点中最短路径长度的最大值
    max_path_temp = max(np.array(list(root_decision.values()))[:, 1])
    # print("max_path_temp  == ",max_path_temp," type",type(max_path_temp))
    max_path_temp = int(max_path_temp)
    max_path = max_path_temp if max_path_temp > -1 else 2
    for node in root_decision:
        if root_decision[node][1] == -1:  # maximal local leader(s)
            root_decision[node] = [root_decision[node][0], max_path, root_decision[node][2]]

    # (2). calculate normalized influence (here, degree k_i) & path length l_i of all nodes (yields result in Fig. 1f in the main text of our paper) #计算所有节点规一化后的度值ki和最短路径li
    node_plot = root_decision.copy()
    for n in G.nodes:
        if n not in node_plot:
            node_plot[n] = [D.nodes[n]['parentnode'], 1, G.degree()[n]]
    root_array = np.array(list(node_plot.values()))
    #     print('degree, path',root_array[:,2],root_array[:,1])
    root_array[root_array[:, 2] <= 1, 1] = 1  # Set l_i=1 for nodes whose degree k_i=1 ###
    degree = get_indicator_rank(root_array[:, 2])
    shortest_path = get_square(root_array[:, 1])
    degree_standard = standard_data(np.array(degree))
    shortest_path_standard = standard_data(np.array(shortest_path))
    multi = degree_standard * shortest_path_standard  # noralized k_i*l_i
    nodeid = list(node_plot.keys())
    multi_dict = {}
    for i in range(len(nodeid)):
        multi_dict[nodeid[i]] = multi[i]
    multi_sort = np.array(sorted(multi_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    multi_sort = np.array([[int(i[0]), i[1]] for i in multi_sort])
    multi_x = [i for i in range(len(multi_sort))]
    # print('Determine centers by muti:',multi_sort[:40])

    # choosing the first-level community centers automatically when there is multi-scale communities
    if auto_choose_centers == True:
        auto_centernum = choose_center(multi_sort)
        center_num = auto_centernum if center_num < auto_centernum else center_num
    if not center_num:
        center_num = len(root_to_node)
    center_dcd = []
    local_cnt = 0
    # for i in multi_sort[:,1]:
    for i in multi_sort[:center_num]:
        if i[1] > 0:
            local_cnt += 1
            center_dcd.append(int(i[0]))
    print("The number of local leaders: " + str(local_cnt))
    # saving related data for visualization  #保存绘图需要的数据
    plot_combination_data = [root_array[:, 2], root_array[:, 1], nodeid, multi_x, multi_sort[:, 1], multi_sort[:, 0],
                             center_dcd]
    plot_process_degree_shortpath_data = [degree, shortest_path, nodeid]


    # (3). For local leaders, record their superior along the hierarchy in the DAG
    for node in root_to_node.keys():
        D.nodes[node]["parentnode"] = root_decision[node][0]
        D.nodes[node]["rootnode"] = D.nodes[node]["parentnode"]
    for node in D.nodes:
        recent_node = []  # prevent loop
        recent_node.append(node)
        flag = 0
        if node in center_dcd:
            D.nodes[node]["rootnode"] = node
        else:
            while D.nodes[node]["rootnode"] not in center_dcd and flag == 0:
                j = D.nodes[node]["rootnode"]
                if j not in recent_node and j != None:
                    recent_node.append(j)
                    D.nodes[node]["rootnode"] = D.nodes[j]["rootnode"]
                else:
                    D.nodes[node]["rootnode"] = None
                    flag = 1

    # (4). get the classes and partition
    y_dcd = []
    y_partition = {}
    for node in D.nodes:
        if D.nodes[node]["rootnode"] == None:
            y_dcd.append(-1)
            y_partition[node] = -1
        else:
            y_dcd.append(D.nodes[node]["rootnode"])
            y_partition[node] = D.nodes[node]["rootnode"]

    end_time = datetime.now()
    stamp = (end_time - start_time).total_seconds() * 1000
    print('Running Time: %d ms' % stamp)

    # print('The number of community  centers: '+str(center_dcd))
    print('The number of community  centers: ' + str(len(plot_combination_data[6])))
    print('The id of the centers are: ' + str(plot_combination_data[6]))
    # print('Modularity of the partition by LS: '+str(community.modularity(y_partition, G)))

    print("The decision graph for determining the number of centers, " +
          "where centers are nodes with both a large influence k_i and path length l_i to other local leaders with a higher influence.")

    from collections import defaultdict

    grouped_dict = defaultdict(list)
    for key, value in y_partition.items():
        grouped_dict[value].append(key)

    # Print partition summary
    if grouped_dict:
        print("Communities (center: members):")
        for center, members in grouped_dict.items():
            print(f"  {center}: {sorted(members)}")

    # just for better visualization, can be safely modified
    if isdraw == True:
        subplot_location = [0.25, 0.55, 0.35, 0.3]
        xlim_start_end = [0.3, 0.7]
        ylim_start_end = [0.7, 0.3]
        font_location = -0.04
        plot_combination(plot_combination_data[0], plot_combination_data[1], plot_combination_data[2],
                         plot_combination_data[3], plot_combination_data[4], plot_combination_data[5],
                         plot_combination_data[6], subplot_location, xlim_start_end, ylim_start_end, font_location,
                         filepath=plot_filepath, dataname=plot_dataname, save=True, show=plot_show)

    print(
        "Note: If multi-scale community structure, which can be common in real networks, is of interest, the number of communities at different level can be explicitly set by some sophisticaed methods or simply by visual inspection for notable gaps in the decision graph. In the default setting, LS alorithm returns community partition at the finest level.")
    return D, center_dcd, y_dcd, y_partition, grouped_dict, plot_combination_data


def LS_degree_communities(
        G,
        center_num=None,
        auto_choose_centers=False,
        maximum_tree=True,
        isdraw=True,
        seed=None,
        self_loop=False,
        plot_filepath="./",
        plot_dataname="LS_default",
        plot_show=False,
):
    """Alias for hierarchical_degree_communities with the same parameters."""
    return hierarchical_degree_communities(
        G,
        center_num=center_num,
        auto_choose_centers=auto_choose_centers,
        maximum_tree=maximum_tree,
        isdraw=isdraw,
        seed=seed,
        self_loop=self_loop,
        plot_filepath=plot_filepath,
        plot_dataname=plot_dataname,
    )


# if __name__ == '__main__':
#     print("### Simple (extreme) example of network where this method does not produce a unique community ###")
#     G=eg.Graph()
#     #G.add_edges_from(EdgeList)
#     # # load the network data
#     seed = 163
#     G.add_edges_from([ (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5) ])  #here is a simple example
#     # G.add_edges_from([(0, 1), (2, 3), (4, 5)])
#     print(G.nodes)
#     print(type(G.nodes))
#     # If you want to use your own dataset, use to read and set label = "id" e.g. eg.read_gml("your dataset",label="id")
#     # G=eg.read_gml("net_SBM_compact_nb_groups_100_block_size_5_p_in_0.8_k_out_8_i_0.gml",label="id")


#     D, center_dcd, y_dcd, y_partition, grouped_dict, plot_combination_data = hierarchical_degree_communities(G, maximum_tree=True, seed=seed)
#     print("Key represents the community center, and Value represents the nodes within the community.")
#     print(grouped_dict)
#     # hierarchical_degree_communities(G, maximum_tree=False, seed=seed)
#     # print('If there is multi-scale community structure, you can type the number of communities:')
#     # nc = int(input())
#     # hierarchical_degree_communities(G, maximum_tree = True, isdraw = False, seed=seed, center_num=nc)
#     # print("Key represents the community center, and Value represents the nodes within the community.")
#     # print(grouped_dict)

#     # # Other examples
#     # print("\n\n  ### Karate Club Network ###")
#     # G=eg.karate_club_graph()
#     # hierarchical_degree_communities(G, maximum_tree=True, seed=seed)
