
import topologies
import networkx as nx

def checkIfAlive(allocation = [], verbose = False, **kwargs):
    #graphs: [networkGraph, taskGraph, energy_list, graphType, networkType]
    if verbose:
        print("Performing network check")
    time = 0
    latency = 0
    nNodes = kwargs['nNodes']
    network_creator = topologies.network_topologies[kwargs['network_creator']]
    nTasks = kwargs['nTasks']
    task_creator = topologies.task_topologies[kwargs['task_creator']]
    energy_list = kwargs['energy_list_sim']
    node_status = kwargs['network_status']
    init_energy = kwargs['init_energy']
    networkGraph = network_creator(**kwargs)
    taskGraph = task_creator(networkGraph, **kwargs)   
    to_remove = []
    for i,node in enumerate(list(networkGraph.nodes())):
        if node.energy < 0.1*init_energy:
            to_remove.append(node)
        elif not node_status[i]:
            to_remove.append(node)
    networkGraph.remove_nodes_from(to_remove)
    if len(networkGraph.nodes()) < 1:
        print("network has 0 nodes left")
        return False
    if not(nx.is_connected(networkGraph)):
        print(f"network is no longer connected: \
                \n   status: {kwargs['network_status']} \
                \n   energies: {kwargs['energy_list_sim']} \
                " )
        return False
    return True



def remove_dead_nodes(graph, energy, energy_only=False, **kwargs):
    to_remove = []
    to_remove_ids = []
    for i,node in enumerate(list(graph.nodes())):
        if node.energy <= 0.1*kwargs['init_energy']:
            to_remove.append(node)
            to_remove_ids.append(i)
        elif not kwargs['network_status'][i] and not energy_only:
            to_remove.append(node)
            to_remove_ids.append(i)
    print("removed nodes:")
    print(to_remove_ids)
    graph.remove_nodes_from(to_remove)
