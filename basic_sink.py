import topologies
import network
import numpy as np
import ns.netanim
from nsga2 import random_assignment
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

nNodes = 10
nTasks = 20
dims = 10
energy = 3

network_creator = topologies.Grid
task_creator = topologies.OneSink
if network_creator == topologies.Grid:
    nNodes = dims ** 2
energy_list = [energy] * nNodes

network_status = [1]*nNodes

settings = {'nNodes': nNodes,
            'network_creator': network_creator,
            'dimx': dims,
            'dimy': dims,
            'deltax' :10,
            'deltay': 10,
            'nTasks': nTasks,
            'task_creator': task_creator,
            'energy_list': energy_list,
            'init_energy': energy,
            'verbose': True,
            'algorithm': 'nsga2',
            'capture_packets': True,
            'enable_errors' : False,
            'error_shape' : 1.0,
            'error_scale' : 3.0,
            'network_status' : network_status,
            'run_number' : 1,
            }

networkGraph = network_creator(**settings)
task_graph = task_creator(networkGraph, **settings)
net = network.Network(networkGraph, **settings)
allocation = random_assignment(networkGraph, task_graph)
print(allocation)
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
node_status = []
latency_list = []
received_list = []
act_list = []
sent_list = []
send_list = []
energy_list = []
time = []

allocation =random_assignment(networkGraph, task_graph)
alloc2 = [(0,[0])]
for i, alloc in enumerate(allocation):
    alloc2.append((i+1,[alloc]))

anime = ns.netanim.AnimationInterface("animation.xml")

ns.core.RngSeedManager.SetRun(1)
ns.core.Simulator.ScheduleDestroy(net.getLatency, latency_list)
ns.core.Simulator.ScheduleDestroy(net.getPackagesSent, sent_list, send_list)
ns.core.Simulator.ScheduleDestroy(net.getPackagesReceived, received_list, act_list)
ns.core.Simulator.ScheduleDestroy(net.getEnergy, energy_list)
def getTime(time = []):
    time.append(ns.core.Simulator.Now().GetSeconds())
ns.core.Simulator.ScheduleDestroy(getTime, time)
ns.core.Simulator.ScheduleDestroy(net.getNodeStatus, node_status)
#ns.core.Simulator.Schedule(ns.core.Time(1), net.sendAllocationMessages)
ns.core.Simulator.Stop(ns.core.Time(ns.core.Seconds(20)))
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()

print(latency_list)
latency = max(latency_list)
print(sent_list)
print(received_list)
missed_packages = sent_list[0] - received_list[0]
percentage = missed_packages/sent_list[0]
#print(f"expected packages: {int(time[0]/5+1)}")
print(f"actually sent: {sent_list}")
print(f"received: {received_list}")
print(f"every sendtask sent: {send_list}")
print(f"act received: {act_list}")
print(f"missed packages:{missed_packages}")
print(f"% missed: {percentage}")
print(f"time running: {time}")
print(f"Node state: {node_status}")
print(f"Latency: {latency}")
latency += latency*percentage
time = np.mean(time)
time -= time*percentage
net.controlTask = 0
net = 0
nodetasks = 0
print(time)


#time, latency, rcv, en = network.evaluate(allocation, **settings)