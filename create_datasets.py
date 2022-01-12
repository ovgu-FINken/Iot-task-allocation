import network
import topologies
import numpy as np
from task import Task
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
import os
import itertools
import time as timer
import tools
import json
import pickle as pck
from datetime import datetime
nNodes = 81
dims = 9
energy = 100
network_creator = topologies.ManHattan
task_creator = None
nTasks = 0
if network_creator == topologies.Grid or network_creator==topologies.ManHattan:
    nNodes = dims**2
if network_creator == topologies.Line:
    dims = nNodes
for static in [False, True]:
    print(f"creating static = {static} mobility data")
    predl = ['perfect', 'target'] if not static else ['perfect']
    for predictor in predl:
        print(f"creating {predictor} predictor data")
        mobl = [0,30,40]
        mobl = mobl if not static else [0]
        for nMob in mobl:
            print(f"creating {nMob} n mobile nodes data")
            mobileNodes = nMob
            nNodes = 81 + mobileNodes
            energy_list = [energy]*nNodes
            network_status = [1]*nNodes
            for i in range(11):
                    print(f"{datetime.now().strftime('%H:%M:%S')}: creating dataset {i}")
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
                             'seed' : 3141 + i*21,
                             'error_shape' : 1.0,
                             'error_scale' : 1.0,
                             'network_status' : network_status,
                             'routing' : True,
                             'static' : static,
                             'run_number' : i,
                             'predictor' : predictor
                             }
                    #ns.core.RngSeedManager.SetSeed(settings['seed'])
                    ns.core.RngSeedManager.SetRun(i)
                    
                    networkGraph = network_creator(**settings)
                    posL = [[float(node.posx), float(node.posy)] for node in networkGraph.nodes()]
                    settings['posList'] = posL

                    net = network.Network(networkGraph, **settings)
                    network.createTasksFromGraph(net)

                    node_data = []
                    energy_data = []
                    prediction_data =[]
                    
                    #ns.core.Simulator.Schedule(ns.core.Seconds(0), net.getPredictions, prediction_data, 30)
                    for j in range(1,100):
                        if settings['static']:
                            ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodeStatus, node_data)
                            ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getEnergy, energy_data)
                        elif settings['predictor'] == 'target':
                            ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodeStatus, node_data)
                            ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getEnergy, energy_data)
                            ns.core.Simulator.Schedule(ns.core.Seconds((j-1)*30), net.getPredictions, prediction_data, 30)
                        elif settings['predictor'] == 'perfect':
                            ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodeStatus, node_data)
                            ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getEnergy, energy_data)
                            ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodeStatus, prediction_data)
                    def getTime(time = []):
                        time.append(ns.core.Simulator.Now().GetSeconds())    
                    time = [] 
                    ns.core.Simulator.ScheduleDestroy(getTime, time)
                    ns.core.Simulator.Run()
                    ns.core.Simulator.Destroy()
                    #print(list(predictions))
                    #print((list(node_data)))
                    #print(list(energy_data))
                    
                    e = list(energy_data)
                    n = list(node_data)
                    s2 = str(nNodes)
                    settings.update({'network_creator' : 'Manhattan'})
                    if settings['static']:
                        s1 = 'static'
                        s3 = ''
                        #with open(f"datasets/{s1}/{s2}/energy_{i}.json",'w') as f:
                        #    json.dump(e, f)
                        s=f"{os.getcwd()}/datasets/{s1}/{s2}/"
                        if (not os.path.exists(s)):
                            os.makedirs(s)
                        with open(f"{s}positions_{i}.json",'w+')as f:
                            json.dump(n, f)
                        with open(f"{s}/settings_{i}.json",'w+')as f:
                            json.dump(settings, f)
                        with open(f"{s}/time_{i}.json",'w+')as f:
                            json.dump(time, f)
                    else:
                        s1 = 'mobile'
                        s3 = settings['predictor']
                        s=f"{os.getcwd()}/datasets/{s1}/{s2}/{s3}"
                        if (not os.path.exists(s)):
                            os.makedirs(s)
                        with open(f"{s}/positions_{i}.json",'w+')as f:
                            json.dump(n, f)
                        with open(f"{s}/settings_{i}.json",'w+')as f:
                            json.dump(settings, f)
                        with open(f"{s}/predictions_{i}.json",'w+')as f:
                            json.dump(prediction_data, f)
                        with open(f"{s}/time_{i}.json",'w+')as f:
                            json.dump(time, f)

                    net.controlTask = 0
                    net.cleanUp()
                    net = 0
