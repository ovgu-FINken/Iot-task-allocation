import networkx as nx
import topologies
import numpy as np
from task import Task



def getCriticalNodes(network, task, settings):
    critical_groups = []
    for nTask in task:
        group = nTask.get_constrained_nodes(network)
        assert len(group), "Empty critical group found"
        critical_groups.append(group)
    
    if settings['network_creator'] == topologies.ManHattan:
        #critical manhattan groups left-right
        for x in range(settings['dimx']):
            pos = x*settings['deltax']+50
            group = [node for node in network.nodes() if node.posx == pos]
            assert len(group), "Empty critical group found"
            critical_groups.append(group)
    
    return critical_groups





#critGroups = getCriticalNodes(network, task, settings)
#print(getCriticalNodes(network, task, settings))

def sum_energy(group):
    energy_sum=0
    for node in group:
        energy_sum += node.energy
    return energy_sum

def update_energy(network, new_energy):
    for en, node in zip(new_energy,network):
        node.update_energy(en)


def calculateGroupFailure(network, task, new_energy, deltaT, settings):
    groups = getCriticalNodes(network, task, settings)
    energy_pools_old = [sum_energy(group) for group in groups]
    #print(energy_pools_old)
    update_energy(network, new_energy)
    energy_pools_new = [sum_energy(group) for group in groups]
    #print(energy_pools_new)
    
    energy_delta = [new-old for new,old in zip(energy_pools_new, energy_pools_old)]
    time_to_failure = [old/delta for old, delta in zip(energy_pools_old, energy_delta)]
    #print(min(time_to_failure))
    #print(deltaT)
    return(min(time_to_failure)*deltaT)


if __name__ == '__main__':
    network_creator = topologies.ManHattan
    task_creator = topologies.TwoTask
    nNodes= 49
    nTasks=7
    dims=7
    energy=100
    energy_list=[energy]*nNodes
    network_status = [1]*nNodes
    settings = {'nNodes' : nNodes,
             'mobileNodeCount' : 0,
             'network_creator' : network_creator,
             'dimx' : dims,
             'dimy' : dims,
             'deltax' :100,
             'deltay': 100,
             'nTasks' : nTasks,
             'task_creator' : task_creator,
             'energy_list' : energy_list ,
             'posList' : [],
             'init_energy' : energy,
             'verbose' : False,
             'capture_packets' : True,
             'pcap_filename' : f"pcap_minimal_network_{nTasks}task",
             'enable_errors' : False,
             'seed' : 5,
             'error_shape' : 1.0,
             'error_scale' : 1.0,
             'network_status' : network_status,
             'routing' : True,
             'static' : True,
             'predictor' : 'perfect'
             }
    network = network_creator(**settings)
    task = task_creator(network, **settings)
    NL = calculateGroupFailure(network, task, [75]*nNodes, 10, settings)
    print(NL)



    




