import network
import topologies
import numpy as np
from task import Task
from nsga2 import random_assignment
import networkx as nx
import ns.core
import ns.network
import ns.applications
import ns.lr_wpan
import ns.mobility
import ns.sixlowpan
import ns.energy
import os
import sys
import itertools
import time as timer
import tools
import json
import pickle as pck
from datetime import datetime
from graph_analyze import calculateGroupFailure

import torch
from torch_geometric.utils.convert import to_networkx, from_networkx

def create_training_batch(nNodes, dims, energy, nTasks, network_creator, task_creator, i):
    energy_list=[energy]*nNodes
    network_status = [1]*nNodes
    if network_creator == topologies.Grid or network_creator==topologies.ManHattan:
        nNodes = dims**2
    if network_creator == topologies.Line:
        dims = nNodes
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
                     'capture_packets' : False,
                     'pcap_filename' : f"pcap_minimal_network_{nTasks}task",
                     'enable_errors' : False,
                     'seed' : 3141 + i*21,
                     'error_shape' : 1.0,
                     'error_scale' : 1.0,
                     'network_status' : network_status,
                     'predictor' : 'perfect',
                     'routing' : True,
                     'static' : True,
                     'run_number' : i,
                     }
                        #ns.core.RngSeedManager.SetSeed(settings['seed'])
    ns.core.RngSeedManager.SetRun(i)
    networkGraph = network_creator(**settings)
    taskGraph = task_creator(networkGraph, **settings)
    posL = [[float(node.posx), float(node.posy)] for node in networkGraph.nodes()]
    settings['posList'] = posL
    net = network.Network(networkGraph, **settings)
    allocation = random_assignment(networkGraph, taskGraph)
    network.createTasksFromGraph(net, taskGraph, allocation, **settings)


    def getTime(time=[]):
        time.append(ns.core.Simulator.Now().GetSeconds())


    node_status = []
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


    ns.core.Simulator.ScheduleDestroy(net.getLatency, latency_list)
    ns.core.Simulator.ScheduleDestroy(net.getPackagesSent, sent_list, send_list, seqNumTx)
    ns.core.Simulator.ScheduleDestroy(net.getPackagesReceived, received_list, act_list, seqNumRx)
    ns.core.Simulator.ScheduleDestroy(net.getEnergy, energy_list)
    ns.core.Simulator.ScheduleDestroy(getTime, time)
    ns.core.Simulator.Stop(ns.core.Time(ns.core.Seconds(100)))
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()
    latency = max(latency_list) if len(latency_list) > 0 else 0
    missed_packages = sent_list[0] - received_list[0]
    percentage = missed_packages/sent_list[0] if sent_list[0] > 0 else 1
    nMissed = 0
    missed_seqNums = []
    for tx in itertools.zip_longest(*seqNumTx,fillvalue=-1):
        found = [False for x in tx]
        for i,a in enumerate(tx):
            for r in seqNumRx:
                if a in r or a == -1:
                    found[i] = True
                    continue
        if not all(found):
            nMissed +=1
    txall = max([len(list(x)) for x in seqNumTx])
    
    missed_perc = nMissed/txall
    NL = calculateGroupFailure(networkGraph,taskGraph,energy_list, time[0], settings)
    usable = True
    if missed_perc >= 1:
        latency = np.inf
        usable = False
    y=[NL,latency, nMissed, missed_perc, missed_packages, percentage]


    for task, node in enumerate(allocation):
        list(networkGraph.nodes)[node].update_task_data(list(taskGraph.nodes)[task])



    for node in list(networkGraph.nodes(data=True)):
        x = []
        for key, value in vars(node[0]).items():
            if not (key == 'pos'):
                x.append(value)
        
        node[1].clear()
        node[1].update({'x' : x})
        node[1].update({'y' : y})

    pyg_graph = from_networkx(networkGraph)
    #print(pyg_graph)
    #print(pyg_graph.x)
    #print(pyg_graph.y)
    net.controlTask = 0
    net.cleanUp()
    net = 0
    return pyg_graph, usable



if __name__ == "__main__":
    nNodes= 81
    dims = 5
    nTasks = 13
    energy = 100
    i = int(sys.argv[1])
    network_creator = topologies.ManHattan
    task_creator = topologies.EncodeDecode
    data, usable = create_training_batch(nNodes, dims, energy, nTasks, network_creator, task_creator, i)
    if i%25 == 0:
        print(f"Processed {i+1} out of 1000 allocations")
        #print(data.y)
    torch.save(data, f"{os.getcwd()}/trainingdata/data{i}.pt")







