import random
import easygraph as eg
from queue import Queue

__all__ = [
    "LPA",
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

def CheckConnect(G, nodes, result):
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
        return CheckConnect(G, [w for w in nodes if w not in seen],result) 
    else :
        result.append(nodes)
        return 

def LPA(G):
    '''Detect community by label propagation algotithm

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
    -------
    nodes : dictionary
      key: serial number of community , value: nodes in the community.

    '''
    i = 0
    label_dict = dict() 
    result_community = dict()
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
    community = [community for label,community in cluster_community.items()]     
    communityx = []   
    for nodes in community:
        CheckConnect(G,nodes,communityx)
    i = 0
    for com in communityx:
        i += 1
        result_community[i] = com
    return result_community

if __name__ == '__main__':
    karate_G = eg.datasets.get_graph_karateclub()
    community = LPA(karate_G)
    print(community)