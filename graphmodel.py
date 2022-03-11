from graph_analyze import calculateGroupFailure
import networkx as nx


def predict(networkGraph, taskGraph, allocation, settings):
    energy_list = [node.energy for node in list(networkGraph.nodes())]
    for t,n in enumerate(allocation):
        node = list(networkGraph.nodes())[n]
        task = list(taskGraph.nodes())[t]
        if task.task_type=='Sending':
            neighbors = networkGraph.neighbors(node)
            for neighbor in neighbors:
                neighbor.energy -= 0.01
            node.energy -= 0.05
        elif task.task_type == 'Processing':
            neighbors = networkGraph.neighbors(node)
            for neighbor in neighbors:
                neighbor.energy -= 0.01
            node.energy -= 0.055
        elif task.task_type == 'Actuating':
            node.energy -= 0.02
    for node in list(networkGraph.nodes):
        node.energy -= 0.01


    NL = calculateGroupFailure(networkGraph,taskGraph,energy_list, 10, settings)
    
    l = 0
    for i, task in enumerate(list(taskGraph.nodes())):
        if task.task_type == 'Sending':
            edgelist = nx.dfs_edges(taskGraph, source = task)
            l1 = 0.1
            for edge in edgelist:
                if edge[0] == task:
                    l1 = 0.1
                else: 
                    source = edge[0]
                    target = edge[1]
                    sourceId = list(taskGraph.nodes()).index(source)
                    targetId = list(taskGraph.nodes()).index(source)
                    source_node_id = allocation[sourceId]
                    target_node_id = allocation[targetId]
                    source_node = list(networkGraph.nodes())[source_node_id]
                    target_node = list(networkGraph.nodes())[target_node_id]
                    hops = len(nx.shortest_path(networkGraph, source_node, target_node))
                    l1 += 0.1*(hops-1)
            l = max(l, l1)

    
    a = max(set(allocation), key=allocation.count)
    
    av = max(1.1-(allocation.count(a)*0.1),0)
    

    #print(NL,l, av)
    return -NL, l, av







            
