import network
import topologies
import numpy as np
from task import Task

import networkx as nx
import ns.core
import ns.network
import ns.point_to_point
import ns.applications
import ns.wifi
import ns.lr_wpan
import ns.mobility
import ns.csma
import ns.internet 
import ns.sixlowpan
import ns.internet_apps
import ns.energy

nNodes = 3
nTasks = 2
dims = 2
energy = 100
network_creator = topologies.Line
task_creator = topologies.TwoTaskWithProcessing
energy_list = [energy]*nNodes

settings = {'nNodes' : nNodes,
         'network_creator' : network_creator,
         'dimx' : dims,
         'dimy' : dims,
         'nTasks' : nTasks,
         'task_creator' : task_creator,
         'energy_list' : energy_list ,
         'init_energy' : energy,
         'verbose' : True,
         'capture_packets' : True,
         'enable_errors' : True,
         'error_shape' : 1.0,
         'error_scale' : 1.0
         }

networkGraph = network_creator(**settings)
print(nx.to_dict_of_lists(networkGraph))
task_graph = task_creator(networkGraph, **settings)
net = network.Network(networkGraph, **settings)
allocation = [0,nNodes-1]


network.createTasksFromGraph(net, task_graph, allocation, **settings)

nodetasks = net.taskApps.Get(0).GetTasks()

n2 = nx.convert_node_labels_to_integers(networkGraph)
edgelist = nx.generate_edgelist(n2, data=False)
edge_export = []
for pair in edgelist:
    nodes = pair.split(' ')
    edge_export.append(int(nodes[0]))
    edge_export.append(int(nodes[1]))
net.controlTask.UpdateGraph(edge_export)

latency_list = []
received_list = []
sent_list = []
energy_list = []
time = []
def getTime(time = []):
    time.append(ns.core.Simulator.Now().GetSeconds())    
ns.core.Simulator.ScheduleDestroy(net.getLatency, latency_list)
ns.core.Simulator.ScheduleDestroy(net.getPackagesSent, sent_list)
ns.core.Simulator.ScheduleDestroy(net.getPackagesReceived, received_list)
ns.core.Simulator.ScheduleDestroy(net.getEnergy, energy_list)
ns.core.Simulator.ScheduleDestroy(getTime, time)
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()


latency = max(latency_list)
print(sent_list)
print(received_list)
missed_packages = sent_list[0] - received_list[0]
percentage = missed_packages/sent_list[0]
print(f"expected packages: {int(time[0]/5+1)*2}")
print(f"actually sent: {sent_list[0]}")
print(f"received: {received_list[0]}")
print(f"missed packages:{missed_packages}") 
print(f"% missed: {percentage}")
print(f"time running: {time}")
latency += latency*percentage
time = np.mean(time)
time -= time*percentage
print(time)
