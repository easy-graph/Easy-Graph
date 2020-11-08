import random
import numpy as np
from queue import Queue
import easygraph as eg
from collections import defaultdict

__all__ = [
    "LPA",
    "SLPA",
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
            # Asynchronous updates. If you want to use synchronous updates, comment the line below
            label_dict[node] = random.choice(labels)
            Next_label_dict[node] = random.choice(labels)
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