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
         'verbose' : False,
         'capture_packets' : True
         }

networkGraph = network_creator(**settings)
task_graph = task_creator(networkGraph, **settings)
net = network.Network(networkGraph, **settings)
allocation = [0,2]


network.createTasksFromGraph(net, task_graph, allocation, **settings)

print(net)
print(list(task_graph))


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
