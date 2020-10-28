import random
import numpy as np
from queue import Queue
import easygraph as eg
from collections import defaultdict

__all__ = [
    "LPA",
    "SLPA",
    "HANP",
]

def SelectLabels(G,node,label_dict):
    adj = G.adj
    count = {}
    for neighbor in adj[node]:
        neighbor_label = label_dict[neighbor]
        count[neighbor_label] = count.get(neighbor_label, 0) + 1
        count_items = sorted(count.items(),key = lambda x: x[1], reverse = True)
    labels = [k for k,v in count_items if v == count_items[0][1]]
    return labels

def estimate_stop_cond(G,label_dict):
    for node in G.nodes:
        if label_dict[node] not in SelectLabels(G,node,label_dict):
            return False
    return True

def SelectLabels_HANP(G,node,label_dict,score_dict,degrees,m,threshod):
    adj = G.adj
    count = defaultdict(float)
    cnt = defaultdict(int)
    for neighbor in adj[node]:
        neighbor_label = label_dict[neighbor]
        cnt[neighbor_label] += 1
        count[neighbor_label] += score_dict[neighbor_label] * (degrees[neighbor] ** m) * adj[node][neighbor].get("weight",1)
    count_items = sorted(count.items(),key = lambda x: x[1], reverse = True)
    labels = [k for k,v in count_items if v == count_items[0][1]]
    # only update node whose number of neighbors sharing the maximal label is less than a certain percentage.
    if cnt[count_items[0][0]] / len(adj[node]) > threshod:
        return [label_dict[node]]
    return labels

def HopAttenuation_Hier(G, node, label_dict, node_dict, distance_dict):
    distance = float("inf")
    Max_distance = 0
    adj = G.adj
    label = label_dict[node]
    ori_node = node_dict[label]
    for _, distancex in distance_dict[ori_node].items():
        Max_distance = max(Max_distance, distancex)
    for neighbor in adj[node]:
        if label_dict[neighbor] == label:
            distance = min(distance, distance_dict[ori_node][neighbor])
    return (1 + distance) / Max_distance

def UpdateScore_Hier(G, node, label_dict, node_dict, distance_dict):
    return 1 - HopAttenuation_Hier(G, node, label_dict, node_dict,distance_dict)

def UpdateScore(G, node, label_dict, score_dict, delta):
    adj = G.adj
    Max_score = 0
    label = label_dict[node]
    for neighbor in adj[node]:
        if label_dict[neighbor] == label:
            Max_score = max(Max_score, score_dict[label_dict[neighbor]])
    return Max_score - delta

def estimate_stop_cond_HANP(G,label_dict,score_dict,degrees,m,threshod):
    for node in G.nodes:
        if label_dict[node] not in SelectLabels_HANP(G,node,label_dict,score_dict,degrees,m,threshod):
            return False
    return True

def MobineNodes(records, G, label_dict, score_dict, node_dict, Next_label_dict, nodes, degrees, distance_dict):
    onerecord = dict()
    for node,label in label_dict.items():
        if label in onerecord:
            onerecord[label].append(node)
        else:
            onerecord[label] = [node]
    records.append(onerecord)
    Gx = eg.Graph()
    label_dictx = dict()
    score_dictx = dict()
    node_dictx = dict()
    nodesx = []
    cnt = 0
    for record_label in onerecord:
        nodesx.append(cnt)
        label_dictx[cnt] = record_label
        score_dictx[record_label] = score_dict[record_label]
        node_dictx[record_label] = cnt
        cnt += 1
    record_labels = list(onerecord.keys())
    i = 0
    edge = dict()
    adj = G.adj
    for i in range(0, len(record_labels)):
        edge[i] =dict()
        for j in range(0, len(record_labels)):
            if i == j: continue
            inodes = onerecord[record_labels[i]]
            jnodes = onerecord[record_labels[j]]
            for unode in inodes:
                for vnode in jnodes:
                    if unode in adj and vnode in adj[unode]:
                        if j not in edge[i]:
                            edge[i][j] = 0
                        edge[i][j] += adj[unode][vnode].get("weight",1)
    for unode in edge:
        for vnode, w in edge[unode].items():
            if unode < vnode:
                Gx.add_edge(unode, vnode, weight = w )
    G = Gx
    label_dict = label_dictx
    score_dict = score_dictx
    node_dict = node_dictx
    Next_label_dict = label_dictx
    nodes = nodesx
    degrees = G.degree()
    distance_dict = eg.Floyd(G)
    return records, G, label_dict, score_dict, node_dict, Next_label_dict, nodes, degrees, distance_dict
    
def ShowRecord(records):
    """
    e.g.
        records : [ {1:[1,2,3,4],2:[5,6,7,8],3:[9],4:[10],5:[11],6:[12]},
                        {2:[0,1,3],3:[2,4,5]},
                            {2:[0,1]} ]

        process :   {1:[1,2,3,4],2:[5,6,7,8],3:[9],4:[10],5:[11],6:[12]} -> 
                        {2:[ [1,2,3,4] + [5,6,7,8] + [10] ], 3:[ [9] + [11] + [12] ]} ->
                            {2:[ ([ [1,2,3,4] + [5,6,7,8] + [10] ]) + ([ [9] + [11] + [12] ] ]) } ->

        return :    {2:[1,2,3,4,5,6,7,8,10,9,11,12]}
    """
    result = dict()
    first = records[0]
    for i in range(1, len(records)):
        keys = list(first.keys())
        onerecord = records[i]
        result = {}
        for label, nodes in onerecord.items():
            for unode in nodes:
                for vnode in first[keys[unode]]:
                    if label not in result:
                        result[label] = []
                    result[label].append(vnode)
        first = result
    return first   

def CheckConnectivity(G, communities):
    result_community = dict()
    community = [list(community) for label,community in communities.items()]     
    communityx = []   
    for nodes in community:
        BFS(G, nodes, communityx)
    i = 0
    for com in communityx:
        i += 1
        result_community[i] = com
    return result_community

def BFS(G, nodes, result):
    # check the nodes in G are connected or not. if not, desperate the nodes into different connected subgraphs.
    if len(nodes) == 0:
        return 
    if len(nodes) == 1:
        result.append(nodes)
        return 
    adj = G.adj
    queue = Queue()
    queue.put(nodes[0])
    seen = set()
    seen.add(nodes[0])
    count = 0
    while(queue.empty()==0):
        vertex = queue.get()
        count += 1
        for w in adj[vertex]:
            if w in nodes and w not in seen:
                queue.put(w)
                seen.add(w)
    if count != len(nodes):
        result.append([w for w in seen])
        return BFS(G, [w for w in nodes if w not in seen],result) 
    else :
        result.append(nodes)
        return 

def LPA(G):
    '''Detect community by label propagation algotithm

    Return the detected communities. But the result is random.

    Each node in the network is initially assigned to its own community. At every iteration,nodes have
    a label that the maximum number of their neighbors have. If there are more than one nodes fit and 
    available, choose a label randomly. Finally, nodes having the same labels are grouped together as 
    communities. In case two or more disconnected groups of nodes have the same label, we run a simple 
    breadth-first search to separate the disconnected communities

    Parameters
    ----------
    G : graph
      A easygraph graph

    Returns
    ----------
    communities : dictionary
      key: serial number of community , value: nodes in the community.

    Examples
    ----------
    >>> LPA(G)

    References
    ----------
    .. [1] Usha Nandini Raghavan, Réka Albert, and Soundar Kumara: 
        Near linear time algorithm to detect community structures in large-scale networks

    '''
    i = 0
    label_dict = dict() 
    cluster_community = dict()
    Next_label_dict = dict()
    nodes = list(G.nodes.keys())
    for node in nodes:
        label_dict[node] = i
        i = i + 1
    loop_count = 0
    while True:
        loop_count += 1
        print ('loop', loop_count)
        random.shuffle(nodes)
        for node in nodes:
            labels = SelectLabels(G,node,label_dict)
            Next_label_dict[node] = random.choice(labels)
            # Asynchronous updates. If you want to use synchronous updates, comment the line below
            label_dict[node] = Next_label_dict[node]
        label_dict = Next_label_dict
        if estimate_stop_cond(G, label_dict) is True:
            print ('complete')
            break
    for node in label_dict.keys():
        label = label_dict[node]
        # print ("label, node", label, node)
        if label not in cluster_community.keys():
            cluster_community[label] = [node]
        else:
            cluster_community[label].append(node) 
    
    result_community = CheckConnectivity(G, cluster_community)
    return result_community

def SLPA(G, T, r):
    '''Detect Overlapping Communities by Speaker-listener Label Propagation Algorithm

    Return the detected Overlapping communities. But the result is random.

    Parameters
    ----------
    G : graph
      A easygraph graph.
    T : int
      The number of iterations, In general, T is set greater than 20, which produces relatively stable outputs.
    r : int
      a threshold between 0 and 1.

    Returns
    -------
    communities : dictionary
      key: serial number of community , value: nodes in the community.

    Examples
    ----------
    >>> SLPA(G,
    ...     T = 20, 
    ...     r = 0.05
    ...     )     

    References
    ----------
    .. [1] Jierui Xie, Boleslaw K. Szymanski, Xiaoming Liu:
        SLPA: Uncovering Overlapping Communities in Social Networks via A Speaker-listener Interaction Dynamic Process

    '''
    nodes = G.nodes
    adj = G.adj
    memory = {i:{i:1} for i in nodes}
    for i in range(0,T):
        listenerslist = list(G.nodes)
        random.shuffle(listenerslist)
        for listener in listenerslist:
            speakerlist = adj[listener]
            if len(speakerlist) == 0:
                continue
            labels = defaultdict(int)
            for speaker in speakerlist:
                # Speaker Rule
                total = float(sum(memory[speaker].values()))
                keys = list(memory[speaker].keys())
                index = np.random.multinomial(1,[freq / total for freq in memory[speaker].values()]).argmax()
                chosen_label = keys[index]
                labels[chosen_label] += 1
            # Listener Rule
            maxlabel = max(labels.items(), key = lambda x:x[1])[0]
            if maxlabel in memory[listener]:
                memory[listener][maxlabel] += 1
            else:
                memory[listener][maxlabel] = 1
    
    for node, labels in memory.items():
        name_list = []
        for label_name, label_number in labels.items():
            if label_number / float(T + 1) < r:
                name_list.append(label_name)
        for name in name_list:
            del labels[name]

    # Find nodes membership
    communities = {}
    for node, labels in memory.items():
        for label in labels:
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])

    # Remove nested communities            
    nestedCommunities = set()
    keys = list(communities.keys())
    for i, label0 in enumerate(keys[:-1]):
        comm0 = communities[label0]
        for label1 in keys[i+1:]:
            comm1 = communities[label1]
            if comm0.issubset(comm1):
                nestedCommunities.add(label0)
            elif comm0.issuperset(comm1):
                nestedCommunities.add(label1)
    for comm in nestedCommunities:
        del communities[comm]

    # Check Connectivity
    result_community = CheckConnectivity(G, communities)
    return result_community

def HANP(G, m, delta, threshod = 1):
    '''Detect community by Hop attenuation & node preference algotithm

    Return the detected communities. But the result is random.

    Implement the basic HANP algorithm and give more freedom through the parameters, e.g., you can use threshod 
    to set the condition for node updating. You can choose geodesic distance as the measure(instead of receiving the current hop scores 
    from the neighborhood and carry out a subtraction) and When an equilibrium is reached, treat newly combined 
    communities as a single node. If you want to use geodesic distance as score measure, see the comment in the middle of code.

    Parameters
    ----------
    G : graph
      A easygraph graph
    m : float
      used to calculate score, when m > 0, more preference is given to node with more neighbors; m < 0, less
    delta : float
      Hop attenuation
    threshod : float 
      Between 0 and 1, only update node whose number of neighbors sharing the maximal label is less than the threshod.
      e.g., threshod == 1 means updating all nodes. 

    Returns
    ----------
    communities : dictionary
      key: serial number of community , value: nodes in the community.

    Examples
    ----------
    >>> HANP(G,
    ...     m = 0.1, 
    ...     delta = 0.05,
    ...     threshod = 1
    ...     )    

    References
    ----------
    .. [1] Ian X. Y. Leung, Pan Hui, Pietro Liò, and Jon Crowcrof: 
        Towards real-time community detection in large networks

    '''
    label_dict = dict()
    score_dict = dict()
    node_dict = dict()
    Next_label_dict = dict()
    cluster_community = dict()
    nodes = list(G.nodes.keys())
    distance_dict = eg.Floyd(G)
    degrees = G.degree()
    records = []
    loop_count = 0
    i = 0
    old_score = 1
    ori_G = G
    for node in nodes:
        label_dict[node] = i
        score_dict[i] = 1
        node_dict[i] = node
        i = i + 1
    while True:
        loop_count += 1
        print ('loop', loop_count)
        random.shuffle(nodes)
        score = 1
        for node in nodes:
            labels = SelectLabels_HANP(G, node, label_dict, score_dict, degrees,m,threshod)
            old_node = label_dict[node]
            Next_label_dict[node] = random.choice(labels)
            # Asynchronous updates. If you want to use synchronous updates, comment the line below
            label_dict[node] = Next_label_dict[node]

            if old_node == Next_label_dict[node]:
                cdelta = 0
            else:
                cdelta = delta
            score_dict[Next_label_dict[node]] = UpdateScore(G, node, label_dict, score_dict, cdelta)
            # If your network are known to be Hierarchical and overlapping communities, uncomment the lines below and comment the 5 lines above
        # if you choose geodesic distance as the measure, uncomment the 2 lines below.
        #     score_dict[Next_label_dict[node]] = UpdateScore_Hier(G, node, label_dict, node_dict, distance_dict)           
        #     score = min(score, score_dict[Next_label_dict[node]])   
        # if you want to treat newly combined communities as a single node, unomment 4 line below.
        # if old_score - score > 1/3:
        #     old_score = score
        #     #  When an equilibrium is reached, treat newly combined communities as a single node.
        #     records, G, label_dict, score_dict, node_dict, Next_label_dict, nodes, degrees, distance_dict = MobineNodes(records, G, label_dict, score_dict, node_dict, Next_label_dict, nodes, degrees, distance_dict)
        label_dict = Next_label_dict
        if estimate_stop_cond_HANP(G,label_dict,score_dict,degrees,m,threshod) is True:
            print ('complete')
            break
    for node in label_dict.keys():
        label = label_dict[node]
        # print ("label, node", label, node)
        if label not in cluster_community.keys():
            cluster_community[label] = [node]
        else:
            cluster_community[label].append(node) 
    #expand the newly combined communities treated as a single node
    # if you treat newly combined communities as a single node, unomment 2 line below.
    # records.append(cluster_community)
    # cluster_community = ShowRecord(records)
    result_community = CheckConnectivity(ori_G, cluster_community)
    return result_community