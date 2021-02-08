import sys
import random
import exceptions
import time
import topologies

import numpy as np
import matplotlib.pyplot as plt
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





class Network:
    def __init__(self, networkGraph = None, positionSettings = {}, mobilitySettings = {}, appSettings= {}, initEnergyJ = 100, verbose = False, **kwargs):
        assert networkGraph is not None, "Need to specify network graph!"
        self.networkGraph = networkGraph
        nodeCount = len(networkGraph.nodes())
        if verbose:
            print(f"Creating network with {nodeCount} nodes")
        self.nodeContainer = self.initNodes(nodeCount, verbose)
        self.emptyNodeContainer = ns.network.NodeContainer()
        #self.mobilityHelper = self.initPositionAndMobility(positionSettings, mobilitySettings)
        self.mobilityHelper = self.buildNetworkFromGraph(networkGraph, mobilitySettings, verbose)
        self.lrWpanHelper = ns.lr_wpan.LrWpanHelper()
        self.lrWpanDeviceContainer = self.initLrWpan(verbose)
        self.energyContainerList = []
        i = 0
        for node in self.networkGraph.nodes():
            self.energyContainerList.append(self.initEnergy(node, i,verbose))
            i += 1
        self.sixLowPanContainer = self.initSixLowPan(verbose)
        self.ipv6Interfaces = self.initIpv6(verbose = verbose, **kwargs)
        self.taskApps = self.InstallTaskApps(verbose = verbose, **kwargs)
        #TODO: Create generic app model and install on all nodes. (Cant add apps to nodes after simstart) 
        self.disableDAD
        self.currentAllocation = None
        if kwargs['capture_packets']:
            self.enablePcap()


    def buildNetworkFromGraph(self, networkGraph, mobilitySettings, verbose):
        if verbose:
            print(f"Building network with {networkGraph} as base graph")
        mobilityHelper = ns.mobility.MobilityHelper()
        listPosAllocator = ns.mobility.ListPositionAllocator()
        for node in networkGraph.nodes():
            listPosAllocator.Add(ns.core.Vector3D(node.posx, node.posy,0))
        mobilityHelper.SetPositionAllocator(listPosAllocator)
        #mobilityModel = mobilitySettings['model']
        mobilityModel = "ns3::ConstantPositionMobilityModel"
        assert mobilityModel == "ns3::ConstantPositionMobilityModel", f"only ConstantPositionMobilityModel supported, not {mobilityModel}"
        mobilityHelper.SetMobilityModel(mobilityModel)
        mobilityHelper.Install(self.nodeContainer)        
        return mobilityHelper

    def initNodes(self, nodeCount : int, verbose):
        if verbose:
            print(f"Initiating {nodeCount} nodes")
        nodeContainer = ns.network.NodeContainer()
        nodeContainer.Create(nodeCount)
        return nodeContainer 

    def initLrWpan(self, PanID = 0, verbose = False):
        lrWpanDeviceContainer = self.lrWpanHelper.Install(self.nodeContainer)
        #self.lrWpanHelper.AssociateToPan(lrWpanDeviceContainer, PanID)
        if verbose:
            print(f"Creating {lrWpanDeviceContainer.GetN()} lr wpan devices")
        for i in range(lrWpanDeviceContainer.GetN()):
            lrWpanDeviceContainer.Get(i).GetCsmaCa().SetUnSlottedCsmaCa()
            lrWpanDeviceContainer.Get(i).GetMac().SetMacMaxFrameRetries(0)
            #tmp = ns.core.BooleanValue(False)
            lrWpanDeviceContainer.Get(i).SetAttribute("UseAcks", ns.core.BooleanValue(False))
            #lrWpanDeviceContainer.Get(i).GetAttribute("UseAcks", tmp)
            #print(tmp)
            #lrWpanDeviceContainer.Get(i).GetCsmaCa().SetMacMaxBE(5)
            #lrWpanDeviceContainer.Get(i).GetCsmaCa().SetMacMinBE(4)
            #lrWpanDeviceContainer.Get(i).GetCsmaCa().SetUnitBackoffPeriod(25)
        return lrWpanDeviceContainer 

    def initEnergy(self, node, nodeIndex, verbose):
        if verbose:
            print(f"Initiating energy for {node}: {node.energy}")
        #networkGraphWithIntLabels = nx.convert_node_labels_to_integers(self.networkGraph)
        liIonEnergySourceHelper = ns.energy.LiIonEnergySourceHelper()
        liIonEnergySourceHelper.Set("LiIonEnergySourceInitialEnergyJ", ns.core.DoubleValue(node.energy))
        energySourceContainer = liIonEnergySourceHelper.Install(self.nodeContainer.Get(nodeIndex))
        energyModel = ns.lr_wpan.LrWpanRadioEnergyModel()
        energyModel.AttachPhy(self.lrWpanDeviceContainer.Get(nodeIndex).GetPhy())
        energyModel.SetEnergySource(energySourceContainer.Get(0))
        energySourceContainer.Get(0).AppendDeviceEnergyModel(energyModel)
        #for i in range(energySourceContainer.GetN()):
        #        print("energy sources after adding")
        #        print(energySourceContainer.Get(i).GetRemainingEnergy())
        return energySourceContainer


    def initSixLowPan(self, verbose):
        if verbose:
            print(f"Initiating sixlowpan on network with {self.lrWpanDeviceContainer.GetN()} devices")
        sixLowPanHelper = ns.sixlowpan.SixLowPanHelper()
        sixLowPanContainer = sixLowPanHelper.Install(self.lrWpanDeviceContainer)
        return sixLowPanContainer 

    def initIpv6(self, prefix = 2001, verbose = False, **unused_settings):
        if verbose:
            print(f"Initiating ipv6 stack with prefix {prefix} on network with {self.nodeContainer.GetN()} devices, assignin to {self.sixLowPanContainer.GetN()} sixlowpan devices")
        ipv6StackHelper = ns.internet.InternetStackHelper()
        ipv6StackHelper.SetIpv4StackInstall(False)
        ipv6StackHelper.Install(self.nodeContainer)
        ipv6address = ns.internet.Ipv6AddressHelper()
        ipv6address.SetBase(ns.network.Ipv6Address(f"{prefix}:1::"), ns.network.Ipv6Prefix(64))
        ipv6interfaces = ipv6address.Assign(self.sixLowPanContainer)
        return ipv6interfaces

    def disableDAD(self):
        for i in range(self.nodeContainer.GetN()):
            icmpv6 = nodes.Get(i).GetObject(ns.internet.Icmpv6L4Protocol.GetTypeId())
            icmpv6.SetAttribute("RetransmissionTime", ns.core.TimeValue(ns.core.Seconds(15)))
            icmpv6.SetAttribute("DelayFirstProbe", ns.core.TimeValue(ns.core.Seconds(15)))
            icmpv6.SetAttribute("MaxMulticastSolicit", ns.core.IntegerValue(0))
            icmpv6.SetAttribute("MaxUnicastSolicit", ns.core.IntegerValue(0))
            icmpv6.SetAttribute("DAD", ns.core.BooleanValue(False))

    
    def InstallTaskApps(self, verbose = False, enable_errors = True, error_scale = 1.0, error_shape=1.0, network_status=[], **unused_settings):
        if verbose:
            print("Installing Task Apps")
        taskHelper = ns.applications.TaskHelper(enable_errors,error_scale,error_shape)
        taskApps = taskHelper.Install(self.nodeContainer)
        #print(f"setting network status: {network_status}")
        for i in range(taskApps.GetN()):
            taskApps.Get(i).SetNetworkStatus(network_status)

        ngraph = nx.convert_node_labels_to_integers(self.networkGraph)
        for node in ngraph:
            nbrs = ngraph.neighbors(node)
            for n in nbrs:
                taskApps.Get(node).AddNeighbour(taskApps.Get(n))

        return taskApps
    
    def InstallTasks(self, taskGraph, allocation):
        return 0   
        #TODO
        

    def enablePcap(self):
        self.lrWpanHelper.EnablePcapAll("wrap-network", True)
        

    def deactivateNode(self, nodeId = 0):
        #l = ns.core.ObjectPtrContainerValue()
        #self.nodeContainer.Get(0).GetAttribute("DeviceList", l)
        #for i in range(l.GetN()):
        #    print(l.Get(i))
        #    print(dir(l.Get(i)))
        
        typeID = ns.core.TypeId.LookupByName('ns3::Ipv6')
        
        self.emptyNodeContainer.Add(self.nodeContainer.Get(nodeId))
       # print(f"Added node with index {node} to empty nodes, nodeID: {self.emptyNodeContainer.Get(0).GetId()}")
        

        for i in range(self.nodeContainer.Get(nodeId).GetObject(typeID).GetNInterfaces()):
            #print(f"NetDevice: {self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).GetDevice()}")
            
            if i ==1:
                #print(f"Underlying net Device: {self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).GetDevice().GetNetDevice()}")
                self.nodeContainer.Get(nodeId).GetObject(typeID).GetInterface(i).SetDown()
                #self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).GetDevice().GetNetDevice().LinkDown()
            #print(f"Interface IsUp: {self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).IsUp()}")
            #print(f"Device IsLinkUp: {self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).GetDevice().IsLinkUp()}")


        

        #typeID0 = ns.core.TypeId.LookupByName('ns3::SixLowPanNetDevice')
        #print(f"sixlowpandevices: {self.nodeContainer.Get(0).GetObject(typeID1)}")
        #typeID2 = ns.core.TypeId.LookupByName('ns3::Ipv6Interface')
        #print(f"sixlowpandevices: {self.nodeContainer.Get(-1).GetObject(typeID1).GetObject(typeID2)}")
    def getEnergy(self, result_list = []):
        for energySourceContainer in self.energyContainerList:
            result_list.append(energySourceContainer.Get(0).GetRemainingEnergy())
        
    
    def getLatency(self, latency_list = []):
        actTaskId = ns.core.TypeId.LookupByName("ns3::ActuatingTask")
        for i in range(self.taskApps.GetN()):
            for task in self.taskApps.Get(i).GetTasks():
                if task.GetTypeId() == actTaskId:
                    latency_list.append(task.GetAverageLatency())
    
    def getNodeStatus(self, node_status = []):
        ctrlTaskId = ns.core.TypeId.LookupByName("ns3::ControlTask")
        for i in range(self.taskApps.GetN()):
            for task in self.taskApps.Get(i).GetTasks():
                if task.GetTypeId() == ctrlTaskId:
                    taskApp = self.taskApps.Get(i)
                    for j in range(self.taskApps.GetN()):
                        status = taskApp.GetNodeState(j)
                        if (status):
                        #GetState returns m_disabled, so 1 means its off
                            node_status.append(0)
                        else:
                            node_status.append(1)
                    return

    def getPackagesSent(self, packages_sent = [], send_sent = []):
        sendTaskId = ns.core.TypeId.LookupByName("ns3::SendTask")
        for i in range(self.taskApps.GetN()):
            for task in self.taskApps.Get(i).GetTasks():
                if task.GetTypeId() == sendTaskId:
                    packages_sent.append(task.GetPackagesSent())
                    send_sent.append(task.GetNSent())
                    return

    def getPackagesReceived(self, packages_received = [], act_received = []):
        sendTaskId = ns.core.TypeId.LookupByName("ns3::ActuatingTask")
        for i in range(self.taskApps.GetN()):
            for task in self.taskApps.Get(i).GetTasks():
                if task.GetTypeId() == sendTaskId:
                    packages_received.append(task.GetPackagesReceived())
                    act_received.append(task.GetNReceived())
                    return
    def sendAllocationMessages(self):
        sendTaskId = ns.core.TypeId.LookupByName("ns3::SendTask")
        for i in range(self.taskApps.GetN()):
            for task in self.taskApps.Get(i).GetTasks():
                if task.GetTypeId() == sendTaskId:
                    task.Send("AllocationMessage")
                    return

    def __str__(self):
        retVal = ""
        for i in range(self.nodeContainer.GetN()):
            mob = self.nodeContainer.Get(i).GetObject(ns.mobility.MobilityModel.GetTypeId())
            device = self.sixLowPanContainer.Get(i)
            retVal += f"Device {i}: \n {mob.GetPosition()} \n {self.ipv6Interfaces.GetAddress(i,0)} \n {self.ipv6Interfaces.GetAddress(i,1)} \n"
        return retVal 

    def cleanUp(self):
        del self.networkGraph
        self.networkGraph = 0
        del self.controlTask
        self.controlTask = 0
        del self.nodeContainer
        self.nodeContainer = 0
        del self.mobilityHelper
        self.mobilityHelper = 0
        del self.lrWpanHelper
        self.lrWpanHelper = 0
        del self.lrWpanDeviceContainer
        self.lrWpanDeviceContainer = 0
        del self.energyContainerList
        self.energyContainerList = 0
        del self.sixLowPanContainer
        self.initSixLowPanContainer = 0
        del self.taskApps
        self.taskApps = 0


def createTasksFromGraph(network, taskGraph, allocation, verbose = False, **unused_settings):
    procTaskFactory = ns.core.ObjectFactory()
    procTaskFactory.SetTypeId("ns3::ProcessingTask")

    sendTaskFactory = ns.core.ObjectFactory()
    sendTaskFactory.SetTypeId("ns3::SendTask")
    #sendTaskFactory.Set ("Interval", ns.core.TimeValue(ns.core.Seconds(5.0)))

    actTaskFactory = ns.core.ObjectFactory()
    actTaskFactory.SetTypeId("ns3::ActuatingTask")

    relayTaskFactory = ns.core.ObjectFactory()
    relayTaskFactory.SetTypeId("ns3::RelayTask")

    controlTaskFactory = ns.core.ObjectFactory()
    controlTaskFactory.SetTypeId("ns3::ControlTask")
    
    controlTask = controlTaskFactory.Create()
    network.controlTask = controlTask
    real_allocation = [(0,[0])]
    for i, alloc in enumerate(allocation):
        real_allocation.append((i+1,[alloc]))
    
    controlTask.SetInitialAllocation(real_allocation)
    network.taskApps.Get(0).AddTask(controlTask)
    #print(real_allocation)
    networkGraph = nx.convert_node_labels_to_integers(network.networkGraph)
    #TODO: include control task in allocation
    taskList=[]
    if verbose:
        print(f"creating {len(taskGraph.nodes())} tasks")
    assert ((len(taskGraph.nodes()))==len(allocation)), f"Task and allocation list length mismatch: {len(taskGraph.nodes())} vs {len(allocation)}"
    try:
        for task, alloc in zip(taskGraph.nodes(), allocation):
            if verbose:
                print(f"allocating {task} to {alloc}")
            if task.task_type == "Sending":
                if verbose:
                    print(f"creating sendtask")
                sendTask = sendTaskFactory.Create()
                sendTask.DoInitialize()
                network.taskApps.Get(alloc).AddTask(sendTask)
                task.node = list(networkGraph.nodes())[alloc]
                taskList.append(sendTask)
            elif task.task_type == "Processing":
                if verbose:
                    print(f"creating proctask")
                procTask = procTaskFactory.Create()
                procTask.DoInitialize()
                network.taskApps.Get(alloc).AddTask(procTask)
                task.node = list(networkGraph.nodes())[alloc]
                taskList.append(procTask)
            elif task.task_type == "Actuating":
                if verbose:
                    print(f"creating acttask")
                actTask = actTaskFactory.Create()
                actTask.DoInitialize()
                network.taskApps.Get(alloc).AddTask(actTask)
                task.node = list(networkGraph.nodes())[alloc]
                taskList.append(actTask)
                if verbose:
                    print(f"created acttask")
            else:
                print(f"UNRECOGNIZED TASK TYPE {task.task_type}")
                raise exceptions.NoValidNodeException
    except Exception as e:
        print(f"Error during task creation: {e}")
        raise e
    if verbose:
        print("Creating task dependencies")
    assert len(taskGraph.nodes) == len(taskList), "taskgraph and task list length mismatch"
    for nxTask, nsTask in zip (taskGraph.nodes(), taskList):
        assert nxTask.node is not None, "Nx task has no node"
        for outTask in nxTask.outputs:
            assert outTask.node is not None, "node is none in pathfinding"
            try:
                if verbose:
                    print(f"calculating dijkstra from {nxTask.node} to {outTask.node}")
                path = nx.shortest_path(networkGraph, source = nxTask.node, target = outTask.node)
                if verbose: 
                    print(f"Path : {path}")
                    print(f"Assigned paired task to id {outTask.taskId-1}, task list length : {len(taskList)}")
                paired_task = taskList[outTask.taskId-1]
                if len(path) <= 2:
                    if verbose:
                        print("tasks assigned direct neighbors or same node")
                    nsTask.AddSuccessor(paired_task)
                    paired_task.AddPredecessor(nsTask)
                else:
                    if verbose:
                        print("Tasks need relay task bridge")
                    relayList = []
                    for pathNode in path[1:-1]:
                        relayTask = relayTaskFactory.Create()
                        relayTask.DoInitialize()
                        if verbose:
                            print(f"Adding relayTask to {pathNode}")
                        network.taskApps.Get(pathNode).AddTask(relayTask)
                        relayList.append(relayTask)
                    for i, relayTask in enumerate(relayList):
                        if i == 0:
                            nsTask.AddSuccessor(relayList[i])
                            relayTask.AddPredecessor(nsTask)
                            if len(relayList) > 1:
                                # more than 1 relay task, successor is next
                                relayTask.AddSuccessor(relayList[i+1])
                            else:
                                # only 1 relay task, successor is the actual receiving task
                                relayTask.AddSuccessor(paired_task)
                                paired_task.AddPredecessor(relayTask)
                        elif (i < len(relayList)-1):
                            #relay task chaining
                            relayTask.AddSuccessor(relayList[i+1])
                            relayTask.AddPredecessor(relayList[i-1])
                        else:
                            #final relay to receiving task
                            relayTask.AddSuccessor(paired_task)
                            relayTask.AddPredecessor(relayList[i-1])
                            paired_task.AddPredecessor(relayTask)
                    for task in relayList:
                        task = 0
                    relayList = 0
            except Exception as e:
                print(f"Error during task dependency creation: {e} ")
                raise e
    for task in taskList:
        task = 0
    taskList = []
    controlTask = 0                
    if verbose:
        print("Finished task creation")


def remove_dead_nodes(graph, energy, energy_only=False, **kwargs):
    to_remove = []
    for i,node in enumerate(list(graph.nodes())):
        if node.energy <= 0.1*kwargs['init_energy']:
            to_remove.append(node)
        elif not kwargs['network_status'][i] and not energy_only:
            to_remove.append(node)
    graph.remove_nodes_from(to_remove)

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
    networkGraph = network_creator(energy_list = energy_list, **kwargs)
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

def evaluate_surrogate(allocation = [], repeat = False, **kwargs):
    #graphs: [networkGraph, taskGraph, energy_list, graphType, networkType]
    verbose = kwargs['verbose']
    if not (checkIfAlive(**kwargs)):
        raise exceptions.NetworkDeadException
    if verbose:
        print(f"Evaluating {allocation}")
        for key,value in kwargs.items():
            print(f"{key} : {value}")
    time = 0
    latency = 0
    nNodes = kwargs['nNodes']
    network_creator = topologies.network_topologies[kwargs['network_creator']]
    nTasks = kwargs['nTasks']
    task_creator = topologies.task_topologies[kwargs['task_creator']]
    energy_list = kwargs['energy_list_eval']
    networkGraph = network_creator(energy_list = kwargs['energy_list_eval'], **kwargs)
    taskGraph = task_creator(networkGraph, **kwargs)   
        
    to_remove = []
    for i,node in enumerate(networkGraph.nodes()):
        if kwargs['energy_list_sim'][i] <= 0.1*kwargs['init_energy']:
            to_remove.append(node)
        elif not kwargs['network_status'][i]:
            to_remove.append(node)
    networkGraph.remove_nodes_from(to_remove)
     
    depleted_indexes = []
    for i,energy in enumerate(kwargs['energy_list_sim']):
        if energy <= 0.1*kwargs['init_energy'] or not kwargs['network_status'][i]:
            depleted_indexes.append((i, energy))
            



    ns.core.LogComponentDisable("SystemMutex", ns.core.LOG_LEVEL_ALL)
    ns.core.LogComponentDisable("Time", ns.core.LOG_LEVEL_ALL)
    if verbose:
        print(f"nodes left: {len(networkGraph.nodes())}")
    try:
        network = Network(networkGraph, **kwargs)
    except Exception as e:
        print(f"Error during network creation: {e}")
        raise e
    if verbose:
        print(f"Creating tasks ")
    createTasksFromGraph(network, taskGraph, allocation, **kwargs)
    #print("Starting Simulation")
    latency_list = []
    received_list = []
    actrcvd = []
    sendsent = []
    sent_list = []
    energy_list = []
    node_status = []
    time = []
    if verbose:
        print("Running sim")
    def getTime(time = []):
        time.append(ns.core.Simulator.Now().GetSeconds())    
    ns.core.RngSeedManager.SetRun(kwargs['run_number'])
    ns.core.Simulator.ScheduleDestroy(network.getLatency, latency_list)
    ns.core.Simulator.ScheduleDestroy(network.getPackagesSent, sent_list, sendsent)
    ns.core.Simulator.ScheduleDestroy(network.getPackagesReceived, received_list, actrcvd)
    ns.core.Simulator.ScheduleDestroy(network.getNodeStatus, node_status)
    ns.core.Simulator.ScheduleDestroy(network.getEnergy, energy_list)
    ns.core.Simulator.ScheduleDestroy(network.getNodeStatus, node_status)
    ns.core.Simulator.Schedule(ns.core.Time(1), network.sendAllocationMessages)    
    ns.core.Simulator.Schedule(ns.core.Time(30), network.sendAllocationMessages)    
    ns.core.Simulator.Stop(ns.core.Time(ns.core.Seconds(300)))
    ns.core.Simulator.ScheduleDestroy(getTime, time)
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()
    #print(latency_list)
    for index, energy in depleted_indexes:
        energy_list.insert(index, energy)
        node_status.insert(index, 0)
    #print(f" time bef: {time}")
    network.cleanUp()
    del network
    network = 0
    
    #the energy this exp started with
    start_energy = kwargs['energy_list_eval']
    #the energy actually left on the nodes
    sim_energy = kwargs['energy_list_sim']
    delta_energy = []
    lifetimes = []
    for old_en, new_en, sim_en in zip(start_energy, energy_list, sim_energy):
        #how much energy was spent?
        deltaT = (old_en-new_en)/time[0]
        delta_energy.append(deltaT)
        if deltaT > 0:
            # how long will the energy last on the actual network?
            lifetimes.append(sim_en/deltaT)
    lifetime = min(lifetimes)

    latency = max(latency_list)
    missed_packages = sent_list[0] - received_list[0]
    percentage = missed_packages/sent_list[0]
    #print(f"sent: {sent_list[0]}")
    #print(f"% missed: {percentage}")
    latency += latency*percentage
    time = np.mean(time)
    time -= time*percentage
    #print(f"missed packages:{missed_packages}") 
    if max(actrcvd) == 0:
        latency = 99999
    received = received_list[0]

    network = 0
    if verbose:
        print("Eval finished")
        print(f"missed packages: {missed_packages} ({percentage}%)")
        print(f"Lifetime: {time}")
        print(f"Latency: {latency}")
        print(f"Energy list: {energy_list}")
        print(f"Node status list: {node_status}")
    #print(f"latency: {latency}")
    #print(energy_list)
    #energy = np.mean(energy_list)
    #print(f"lifetime: {time}")
    #for some reason the network doesnt get deleted corretyl if this isnt done TODO

    return -lifetime, latency, -received, energy_list, node_status, missed_packages
    

def evaluate(allocation = [], repeat = False, **kwargs):
    #graphs: [networkGraph, taskGraph, energy_list, graphType, networkType]
    verbose = kwargs['verbose']
    if not (checkIfAlive(**kwargs)):
        raise exceptions.NetworkDeadException
    if verbose:
        print(f"Evaluating {allocation}")
        for key,value in kwargs.items():
            print(f"{key} : {value}")
    time = 0
    latency = 0
    nNodes = kwargs['nNodes']
    network_creator = topologies.network_topologies[kwargs['network_creator']]
    nTasks = kwargs['nTasks']
    task_creator = topologies.task_topologies[kwargs['task_creator']]
    energy_list = kwargs['energy_list_sim']
    networkGraph = network_creator(energy_list = energy_list, **kwargs)
    taskGraph = task_creator(networkGraph, **kwargs)   
    #print(f"enlist before eval: {energy_list}")
    to_remove = []
    for node in networkGraph.nodes():
        if node.energy <= 0.1*kwargs['init_energy']:
            to_remove.append(node)
    networkGraph.remove_nodes_from(to_remove)
     
    depleted_indexes = []
    for i,energy in enumerate(energy_list):
        if energy <= 0.1*kwargs['init_energy']:
            depleted_indexes.append((i, energy))
    ns.core.LogComponentDisable("SystemMutex", ns.core.LOG_LEVEL_ALL)
    ns.core.LogComponentDisable("Time", ns.core.LOG_LEVEL_ALL)
    if verbose:
        print(f"nodes left: {len(networkGraph.nodes())}")
    try:
        network = Network(networkGraph, **kwargs)
    except Exception as e:
        print(f"Error during network creation: {e}")
        raise e
    if verbose:
        print(f"Creating tasks ")
    createTasksFromGraph(network, taskGraph, allocation, **kwargs)
    #print("Starting Simulation")
    latency_list = []
    received_list = []
    actrcvd = []
    sendsent = []
    sent_list = []
    energy_list = []
    node_status = []
    time = []
    if verbose:
        print("Running sim")
    def getTime(time = []):
        time.append(ns.core.Simulator.Now().GetSeconds())    
    ns.core.RngSeedManager.SetRun(kwargs['run_number'])
    ns.core.Simulator.ScheduleDestroy(network.getLatency, latency_list)
    ns.core.Simulator.ScheduleDestroy(network.getPackagesSent, sent_list, sendsent)
    ns.core.Simulator.ScheduleDestroy(network.getPackagesReceived, received_list, actrcvd)
    ns.core.Simulator.ScheduleDestroy(network.getNodeStatus, node_status)
    ns.core.Simulator.ScheduleDestroy(network.getEnergy, energy_list)
    ns.core.Simulator.ScheduleDestroy(network.getNodeStatus, node_status)
    ns.core.Simulator.Schedule(ns.core.Time(0.5), network.sendAllocationMessages)    
    if len(kwargs['next_alloc']) > 0 and repeat:
        ns.core.Simulator.Stop(ns.core.Time(ns.core.Seconds(20)))    
    ns.core.Simulator.ScheduleDestroy(getTime, time)
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()
    #print(latency_list)
    time1 = []
    latency1 = []
    received1 = []
    energy_list1 = []
    node_status1 = []
    for index, energy in depleted_indexes:
        energy_list.insert(index, energy)
        node_status.insert(index, False)
    #print(f" time bef: {time}")
    network.cleanUp()
    del network
    network = 0
    if repeat:
        if np.mean(time) >= 20 and len(kwargs['next_alloc']) >0 :
            kwargs.update({'energy_list_sim' : energy_list})
            kwargs.update({'network_status' : node_status})
            if verbose:
                print("starting repeat run")
            if not (checkIfAlive(**kwargs)):
                raise exceptions.NetworkDeadException
            try:
                time1, latency1, received1, energy_list1, node_status1, missed1 = evaluate(kwargs['next_alloc'], repeat=False, **kwargs)
            except Exception as e:
                print(f"Error during repeat run: {e}")
                raise e
        


    latency = max(latency_list)
    missed_packages = sent_list[0] - received_list[0]
    percentage = missed_packages/sent_list[0]
    #print(f"sent: {sent_list[0]}")
    #print(f"% missed: {percentage}")
    latency += latency*percentage
    time = np.mean(time)
    time -= time*percentage
    #print(f"missed packages:{missed_packages}") 
    if max(actrcvd) == 0:
        latency = 99999
    received = received_list[0]
        
    if repeat and np.mean(time) >= 20:
        latency =max(latency,latency1)
        time = time + time1
        received += received1
        energy_list = energy_list1
        node_status = node_status1
        missed_packages += missed1

    network = 0
    if verbose:
        print(f"Eval finished - repeat? {repeat}")
        print(f"missed packages: {missed_packages} ({percentage}%)")
        print(f"Lifetime: {time}")
        print(f"Latency: {latency}")
        print(f"Energy list: {energy_list}, len: {len(energy_list)}")
        print(f"Node status list: {node_status}, len: {len(energy_list)}")
    #print(f"latency: {latency}")
    #print(energy_list)
    #energy = np.mean(energy_list)
    #print(f"lifetime: {time}")
    #for some reason the network doesnt get deleted corretyl if this isnt done TODO

    return -time, latency, -received, energy_list, node_status, missed_packages

if __name__ == '__main__':


    cmd = ns.core.CommandLine()
    cmd.verbose = "True"
    cmd.nWifi = 2
    cmd.tracing = "True"

    cmd.AddValue("nWifi", "Number of WSN Nodes")
    cmd.AddValue("verbose", "Tell echo applications to log if true")
    cmd.AddValue("tracing", "Enable pcap tracing")

    cmd.Parse(sys.argv)

    verbose = cmd.verbose
    tracing = cmd.tracing
    #networkGraph = Line(nNodes)
    
    nNodes = 20
    nTasks = 20
    dims = 9
    energy = 3
    network_creator = topologies.Line

    if network_creator == topologies.Grid:
        nNodes = dims**2
    energy_list = [energy]*nNodes

    nWifi = int(cmd.nWifi)
    a = []
    allocation = [x for x in range(nNodes)]
    settings = {'nNodes' : nNodes,
                'network_creator' : network_creator,
                'dimx' : dims,
                'dimy' : dims,
                'nTasks' : nTasks,
                'task_creator' : topologies.TwoTaskWithProcessing,
                'energy_list' : energy_list,
                'init_energy' : energy,
                'verbose' : False,
                'algorithm' : 'nsga2'
                }
    
    allocation = [x for x in range(nNodes)]
    for i in range(2):
        time, latency, rcv, en = evaluate(allocation, **settings) 
    #time, latency, rcv, en = evaluate(allocation, **settings) 
    print(allocation)
    print(time)
    print(latency)
    #print(rcv)
    #print(en)
