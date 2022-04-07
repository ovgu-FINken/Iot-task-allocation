import network
import topologies
import numpy as np
from task import Task
from nsga2 import random_assignment
import networkx as nx
import ns.core
import os
import itertools
import time as timer
import tools
import json
import pickle as pck
from datetime import datetime
nNodes = 49
dims = 7
energy = 1000
network_creator = topologies.ManHattan
task_creator = None
nTasks = 10
if network_creator == topologies.Grid or network_creator==topologies.ManHattan:
    nNodes = dims**2
if network_creator == topologies.Line:
    dims = nNodes
for static in [True, False]:
    print(f"creating static = {static} mobility data")
    predl = ['perfect'] if not static else ['perfect']
    for predictor in predl:
        print(f"creating {predictor} predictor data")
        mobl = [20,40]
        mobl = mobl if not static else [0]
        for nMob in mobl:
            print(f"creating {nMob} n mobile nodes data")
            mobileNodes = nMob
            nNodes = 49 + mobileNodes
            energy_list = [energy]*nNodes
            network_status = [[1,0]]*nNodes
            for error_rate in ['low', 'high', 'zero']:
                if error_rate == 'zero':
                    e_enabled=False
                if error_rate == 'low':
                    e_enabled=True
                    e_shape = 4
                    e_scale = 1000
                    e_dur_shape = 1
                    e_dur_scale = 120
                if error_rate == 'high':
                    e_enabled=True
                    e_shape = 4
                    e_scale = 500
                    e_dur_shape = 1
                    e_dur_scale = 240
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
                                 'broadcast_status': [0]*nNodes,
                                 'posList' : [],
                                 'init_energy' : energy,
                                 'verbose' : False,
                                 'capture_packets' : False,
                                 'pcap_filename' : f"pcap_minimal_network_{nTasks}task",
                                 'enable_errors' : e_enabled,
                                 'seed' : 3141 + i*21,
                                 'error_rate' : error_rate,
                                 'error_shape' : e_shape,
                                 'error_scale' : e_scale,
                                 'error_dur_shape' : e_dur_shape,
                                 'error_dur_scale' : e_dur_scale,
                                 'network_status' : network_status,
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
                        network.createTasksFromGraph(net, **settings)

                        pos_data = []
                        node_status = []
                        broadcast_data = []
                        energy_data = []
                        prediction_data =[]
                        #ns.core.Simulator.Schedule(ns.core.Seconds(0), net.getPredictions, prediction_data, 30)
                        for j in range(1,40):
                            if settings['static']:
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodeStatus, node_status)
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getEnergy, energy_data)
                            elif settings['predictor'] == 'nn':
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodeStatus, node_status)
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodePositions, pos_data)
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodeLastBroadcast, broadcast_data)
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getEnergy, energy_data)
                                ns.core.Simulator.Schedule(ns.core.Seconds((j-1)*30), net.getPredictedPositions, prediction_data, 30)
                            elif settings['predictor'] == 'perfect':
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodeStatus, node_status)
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodePositions, pos_data)
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodeLastBroadcast, broadcast_data)
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getEnergy, energy_data)
                                ns.core.Simulator.Schedule(ns.core.Seconds(j*30), net.getNodePositions, prediction_data)
                        def getTime(time = []):
                            time.append(ns.core.Simulator.Now().GetSeconds())    
                        time = [] 
                        ns.core.Simulator.ScheduleDestroy(getTime, time)
                        ns.core.Simulator.Stop(ns.core.Time(ns.core.Seconds(1300)))
                        ns.core.Simulator.Run()
                        ns.core.Simulator.Destroy()
                    
                        e = list(energy_data)
                        n = list(node_status)
                        p = list(pos_data)
                        b = list(broadcast_data)
                        pred = list(prediction_data)
                        s2 = str(nNodes)
                        print(settings)
                        s4 = settings['error_rate']
                        settings.update({'network_creator' : 'Manhattan'})
                        if settings['static']:
                            s1 = 'static'
                            s3 = ''
                            s=f"{os.getcwd()}/datasets/{s1}/{s2}/{s4}/"
                            if (not os.path.exists(s)):
                                os.makedirs(s)
                            with open(f"{s}energy_{i}.json",'w') as f:
                                json.dump(e, f)
                            with open(f"{s}status_{i}.json",'w+')as f:
                                json.dump(n, f)
                            with open(f"{s}positions_{i}.json",'w+')as f:
                                json.dump(p, f)
                            with open(f"{s}/settings_{i}.json",'w+')as f:
                                json.dump(settings, f)
                            with open(f"{s}/time_{i}.json",'w+')as f:
                                json.dump(time, f)
                        else:
                            s1 = 'mobile'
                            s3 = settings['predictor']
                            s=f"{os.getcwd()}/datasets/{s1}/{s2}/{s3}/{s4}/"
                            if (not os.path.exists(s)):
                                os.makedirs(s)
                            with open(f"{s}energy_{i}.json",'w') as f:
                                json.dump(e, f)
                            with open(f"{s}status_{i}.json",'w+')as f:
                                json.dump(n, f)
                            with open(f"{s}positions_{i}.json",'w+')as f:
                                json.dump(p, f)
                            with open(f"{s}broadcast_{i}.json",'w+')as f:
                                json.dump(b, f)
                            with open(f"{s}predictions_{i}.json",'w+')as f:
                                json.dump(pred, f)
                            with open(f"{s}/settings_{i}.json",'w+')as f:
                                json.dump(settings, f)
                            with open(f"{s}/time_{i}.json",'w+')as f:
                                json.dump(time, f)

                        net.controlTask = 0
                        net.cleanUp()
                        net = 0
