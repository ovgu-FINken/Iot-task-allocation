import network
import topologies
import numpy as np
from task import Task
from nsga2 import random_assignment
import networkx as nx
import ns.core
import ns.network
import ns.applications
import ns.mobility
import ns.energy
import ns.wifi
import ns.internet
import ns.aodv
import itertools
import time as timer

start = timer.time()
nNodes = 4
mobileNodes = 0
nTasks = 3
dims = 2
#nTasks = dims*2-1
energy = 100
network_creator = topologies.ManHattan
#task_creator = topologies.TwoTask
task_creator = topologies.OneSink
#task_creator = None
#nTasks = 0
if network_creator == topologies.Grid or network_creator==topologies.ManHattan:
    nNodes = dims**2
if network_creator == topologies.Line:
    dims = nNodes
nNodes = nNodes + mobileNodes
energy_list = [energy]*nNodes
network_status = [[1,0]]*nNodes
settings = {'nNodes' : nNodes,
         'mobileNodeCount' : mobileNodes,
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
         'capture_packets' : False,
         'pcap_filename' : f"pcap_minimal_network_{nTasks}task",
         'enable_errors' : False,
         'seed' : 5,
         'error_shape' : 10000.0,
         'error_scale' : 1000000.0,
         'broadcast_status': [ns.core.Time(ns.core.Seconds(0))]*nNodes,
         'network_status' : network_status,
         'static' : True,
         'predictor' : 'perfect'
         }


#**Network setup

networkGraph = network_creator(**settings)
task_graph = task_creator(networkGraph, **settings)
#task_graph = None
#print(len(task_graph.nodes))
posL = [[node.posx, node.posy] for node in networkGraph.nodes()]
settings['posList'] = posL
net = network.Network(networkGraph, **settings)
#allocation = list(range(nTasks-1))
#allocation.append(nNodes-1)
#allocation = [0]*(nTasks-1)

#print(len(networkGraph.nodes()))

#right and up:
allocation = list(range(dims-1))
for i in range(dims):
    allocation.append(dims-1+i*dims)

allocation = random_assignment(networkGraph, task_graph)
print(allocation)
#allocation=[6,6,12,18,24]
#allocation=[20,14,18,13,13]
#print("alloc:")
#print(allocation)
network.createTasksFromGraph(net, task_graph, allocation, **settings)

#odetasks = net.taskApps.Get(0).GetTasks()

def updateGraph(nGraph, net):
    n2 = nx.convert_node_labels_to_integers(nGraph)
    edgelist = nx.generate_edgelist(n2, data=False)
    edge_export = []
    for pair in edgelist:
        nodes = pair.split(' ')
        edge_export.append(int(nodes[0]))
        edge_export.append(int(nodes[1]))
    net.controlTask.UpdateGraph(edge_export)

#net.controlTask.UpdateGraph()
node_status = []
node_pos = []
node_broadcast = []
latency_list = []
received_list = []
seqNumTx=[]
seqNumRx=[]
act_list = []
sent_list = []
send_list = []
energy_list = []
time = []
predictions=[]
#allocation2 =random_assignment(networkGraph, task_graph)
#allocation = [0,80]
#alloc2 = [(0,[0])]
#for i, alloc in enumerate(allocation2):
#    alloc2.append((i+1,[alloc]))


#alloc3 = [(0,[0])]
#for i, alloc in enumerate(allocation):
#    alloc3.append((i+1,[alloc]))

state_list = []
def getTime(time = []):
    time.append(ns.core.Simulator.Now().GetSeconds())    
ns.core.RngSeedManager.SetRun(1)
#ns.core.Simulator.Schedule(ns.core.Seconds(1), net.getNodeStatus, node_status)
#ns.core.Simulator.Schedule(ns.core.Seconds(0),net.getPackagesSent, sent_list, send_list)
#ns.core.Simulator.Schedule(ns.core.Seconds(0),net.getPackagesReceived, received_list, act_list)
#ns.core.Simulator.Schedule(ns.core.Seconds(0), net.controlTask.Reallocate, alloc2, net.taskApps)
#ns.core.Simulator.Schedule(ns.core.Seconds(0), net.controlTask.Reallocate, alloc3, net.taskApps)
#ns.core.Simulator.Schedule(ns.core.Seconds(5), net.getPredictions, predictions, 10)
#ns.core.Simulator.Schedule(ns.core.Seconds(5), net.saveState, state_list)
#ns.core.Simulator.ScheduleDestroy(net.getLatency, latency_list)
#ns.core.Simulator.ScheduleDestroy(net.getPackagesSent, sent_list, send_list, seqNumTx)
#ns.core.Simulator.ScheduleDestroy(net.getPackagesReceived, received_list, act_list, seqNumRx)
#ns.core.Simulator.ScheduleDestroy(net.getEnergy, energy_list)
#ns.core.Simulator.ScheduleDestroy(getTime, time)
#ns.core.Simulator.ScheduleDestroy(net.getNodeStatus, node_status)
#ns.core.Simulator.ScheduleDestroy(net.getNodePositions, node_pos)
#ns.core.Simulator.ScheduleDestroy(net.getNodeLastBroadcast, node_broadcast)
#ns.core.Simulator.Schedule(ns.core.Time(1), net.sendAllocationMessages)    
#ns.core.Simulator.Stop(ns.core.Time(ns.core.Seconds(6)))
t1 = timer.time()
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()
#print(state_list)
#print(node_status)
#print()
#print()
#print(f"time elapsed: {timer.time()-t1} for {time} sim seconds")
#print()
#print()
#print(latency_list)
#print(energy_list)
#latency = max(latency_list) if len(latency_list) > 0 else 0
#print(sent_list)
#print(received_list)
#print(f"actually sent: {sent_list}")
#print(f"received: {received_list}")
#print(f"every sendtask sent: {send_list}")
#print(f"act received: {act_list}")
#missed_packages = sent_list[0] - received_list[0]
#percentage = missed_packages/sent_list[0] if sent_list[0] > 0 else 1
#print(f"missed packages:{missed_packages}, {percentage}% of {sent_list[0]}") 
##print(f"Seqnums: {list(seqNumTx)}")
##for x in seqNumTx:
##    print(list(x))
##print(f"Seqnums: {list(seqNumRx)}")
##for x in seqNumRx:
##    print(list(x))
#
#nMissed = 0
#
#
#
#
#for tx in itertools.zip_longest(*seqNumTx,fillvalue=-1):
#    found = [False for x in tx]
#    for i,a in enumerate(tx):
#        for r in seqNumRx:
#            if a in r or a == -1:
#                found[i] = True
#                continue
#    if not all(found):
#        nMissed +=1
#
#print("missed ids:")
#print(nMissed)
#print("out of:")
#txall = max([len(list(x)) for x in seqNumTx])
#print(txall)
#
#
#print(f"% missed: {1.0 - nMissed/txall}")
#print(f"time running: {time}")
#print(f"Node state: {node_status}")
#print(f"Latency: {latency}")
##latency += latency*percentage
##time = np.mean(time)
##time -= time*percentage
#net.controlTask = 0
#net = 0
#nodetasks = 0
##print(time)
#print(predictions)
#print(node_status)
#print(node_broadcast)
#print(f"total time elapsed: {timer.time() -start}")
