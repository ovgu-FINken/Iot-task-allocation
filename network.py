import sys
import networkx as nx
from itertools import combinations
import numpy.linalg as la
import numpy as np
import random
import matplotlib.pyplot as plt
import exceptions
from task import Task 
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

class GraphNode:
    def __init__(self, posx = 0, posy = 0 , energy = 0):
        self.posx = posx
        self.posy = posy
        self.energy = energy
        self.pos = np.array([posx,posy])
    
    def __str__(self):
        return(f"{self.pos} \n {self.energy}")

def Grid(dimx = 10,dimy = 10, deltax=100, deltay=100, energy=100):
    "Create a  grid network with a 4-neighborhood"
    x=0
    y=0
    G=nx.OrderedGraph()
    for i in range(dimx):
        for j in range(dimy):
          #we want to increase x with j and i with j (fill rows first)
          node = GraphNode(x+j*deltax,y+i*deltay, energy)
          G.add_node(GraphNode(x+j*deltax,y+i*deltay, energy), pos=node.pos, energy=node.energy)
    for node1,node2 in combinations(G.nodes(),2):
        dist = la.norm(node1.pos - node2.pos)
        if dist <= 100:
            G.add_edge(node1,node2)
    return G

def Line(nodeCount = 25, deltax = 100, energy = [1000]*25):
    G= nx.OrderedGraph()
    for i in range(nodeCount):
        node = GraphNode(i*deltax,0,energy[i])
        G.add_node(node, pos=node.pos, energy=node.energy)
    for node1,node2 in combinations(G.nodes(),2):
        dist = la.norm(node1.pos - node2.pos)
        if dist <= 100:
            G.add_edge(node1,node2)
    return G



def TwoTask(networkGraph, deltax = 100):
    #reset global taskId counter
    Task.taskId = 1
    n = len(networkGraph.nodes())
    G = nx.OrderedDiGraph()
    first_constraint = {'location' : np.array([np.array([-1,1]),np.array([-1,1])])}
    task1 = Task(first_constraint)
    G.add_node(task1)
    second_constraint = {'location' : np.array([np.array([(n-1)*deltax-1,(n-1)*deltax+1]),np.array([-1,1])])}
    task2 = Task(second_constraint)
    G.add_node(task2)
    G.add_edge(task1,task2)
    for task in G.nodes():
        task.set_topology(G)
    return G
    
def TwoTaskWithProcessing(networkGraph = None, nTasks=0, deltax = 100):
    #reset global taskId counter
    assert networkGraph is not None, "Network Graph for Task Gen is None"
    if nTasks == 0:
        nTasks = len(networkGraph.nodes())
    Task.taskId = 1
    n = len(networkGraph.nodes())
    G = nx.OrderedDiGraph()
    first_constraint = {'location' : np.array([np.array([-1,1]),np.array([-1,1])])}
    task1 = Task(first_constraint)
    G.add_node(task1)
    for i in range(nTasks-2):
        G.add_node(Task())
    second_constraint = {'location' : np.array([np.array([(n-1)*deltax-1,(n-1)*deltax+1]),np.array([-1,1])])}
    task2 = Task(second_constraint)
    G.add_node(task2)
    
    for i, node in enumerate(list(G.nodes())):
        if i == 0:
            continue
        G.add_edge(list(G.nodes())[i-1], node)
    for task in G.nodes():
        task.set_topology(G)
    #for x in list(G.nodes()):
    #    print(x.taskId)
    return G


class Network:
    def __init__(self, networkGraph = None,positionSettings = {}, mobilitySettings = {}, appSettings= {}, initEnergyJ = 100):
        assert networkGraph is not None, "Need to specify network graph!"
        self.networkGraph = networkGraph
        nodeCount = len(networkGraph.nodes())
        self.nodeContainer = self.initNodes(nodeCount)
        self.emptyNodeContainer = ns.network.NodeContainer()
        #self.mobilityHelper = self.initPositionAndMobility(positionSettings, mobilitySettings)
        self.mobilityHelper = self.buildNetworkFromGraph(networkGraph, mobilitySettings)
        self.lrWpanHelper = ns.lr_wpan.LrWpanHelper()
        self.lrWpanDeviceContainer = self.initLrWpan()
        self.energyContainerList = []
        i = 0
        for node in self.networkGraph.nodes():
            self.energyContainerList.append(self.initEnergy(node, i))
            i += 1
        self.sixLowPanContainer = self.initSixLowPan()
        self.ipv6Interfaces = self.initIpv6()
        self.taskApps = self.InstallTaskApps()
        #TODO: Create generic app model and install on all nodes. (Cant add apps to nodes after simstart) 
        self.currentAllocation = None
        #self.enablePcap()


    def buildNetworkFromGraph(self, networkGraph, mobilitySettings):
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

    def initNodes(self, nodeCount : int):
        nodeContainer = ns.network.NodeContainer()
        nodeContainer.Create(nodeCount)
        return nodeContainer 

    def initLrWpan(self, PanID = 0):
        lrWpanDeviceContainer = self.lrWpanHelper.Install(self.nodeContainer)
        self.lrWpanHelper.AssociateToPan(lrWpanDeviceContainer, PanID)
        for i in range(lrWpanDeviceContainer.GetN()):
            lrWpanDeviceContainer.Get(i).GetCsmaCa().SetUnSlottedCsmaCa()
            lrWpanDeviceContainer.Get(i).GetMac().SetMacMaxFrameRetries(0)
            lrWpanDeviceContainer.Get(i).GetCsmaCa().SetMacMaxBE(10)
            #lrwpandevicecontainer.Get(i).GetCsmaCa().SetMacMinBE(5)
            #lrwpandevicecontainer.Get(i).GetCsmaCa().SetUnitBackoffPeriod(255)
        return lrWpanDeviceContainer 

    def initEnergy(self, node, nodeIndex):
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


    def initSixLowPan(self):
        sixLowPanHelper = ns.sixlowpan.SixLowPanHelper()
        sixLowPanContainer = sixLowPanHelper.Install(self.lrWpanDeviceContainer)
        return sixLowPanContainer 

    def initIpv6(self):
        ipv6StackHelper = ns.internet.InternetStackHelper()
        ipv6StackHelper.SetIpv4StackInstall(False)
        ipv6StackHelper.Install(self.nodeContainer)
        ipv6address = ns.internet.Ipv6AddressHelper()
        ipv6address.SetBase(ns.network.Ipv6Address("2001:1::"), ns.network.Ipv6Prefix(64))
        ipv6interfaces = ipv6address.Assign(self.sixLowPanContainer)
        return ipv6interfaces

    
    
    def InstallTaskApps(self):
        taskHelper = ns.applications.TaskHelper()
        taskApps = taskHelper.Install(self.nodeContainer)
        return taskApps
    
    def InstallTasks(self, taskGraph, allocation):
        return 0   
        #TODO
        

    def enablePcap(self):
        self.lrWpanHelper.EnablePcapAll("network-wrapper", True)
        

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
    
    def cleanUp(self):
        self.networkGraph = 0
        self.nodeContainer = 0
        self.emptyNodeContainer = 0
        self.mobilityHelper = 0
        self.lrWpanHelper = 0
        self.lrWpanDeviceContainer = 0
        self.energyContainerList = 0
        self.sixLowPanContainer = 0
        self.ipv6Interfaces = 0
        self.taskApps = 0
        self.currentAllocation = None


    def __str__(self):
        retVal = ""
        for i in range(self.nodeContainer.GetN()):
            mob = self.nodeContainer.Get(i).GetObject(ns.mobility.MobilityModel.GetTypeId())
            device = self.sixLowPanContainer.Get(i)
            retVal += f"Device {i}: \n {mob.GetPosition()} \n {self.ipv6Interfaces.GetAddress(i,0)} \n {self.ipv6Interfaces.GetAddress(i,1)} \n"
        return retVal 


def createTasksFromGraph(network, taskGraph, allocation):
    procTaskFactory = ns.core.ObjectFactory()
    procTaskFactory.SetTypeId("ns3::ProcessingTask")

    sendTaskFactory = ns.core.ObjectFactory()
    sendTaskFactory.SetTypeId("ns3::SendTask")
    sendTaskFactory.Set ("Interval", ns.core.TimeValue(ns.core.Seconds(5.0)))

    actTaskFactory = ns.core.ObjectFactory()
    actTaskFactory.SetTypeId("ns3::ActuatingTask")
    
    relayTaskFactory = ns.core.ObjectFactory()
    relayTaskFactory.SetTypeId("ns3::RelayTask")

    controlTaskFactory = ns.core.ObjectFactory()
    controlTaskFactory.SetTypeId("ns3::ControlTask")
    
    controlTask = controlTaskFactory.Create()
    real_allocation = [(0,[0])]
    for i, alloc in enumerate(allocation):
        real_allocation.append((i+1,[alloc]))
    
    controlTask.SetInitialAllocation(real_allocation)
    network.taskApps.Get(0).AddTask(controlTask)
    #print(real_allocation)
    networkGraph = nx.convert_node_labels_to_integers(network.networkGraph)
    #TODO: include control task in allocation
    taskList=[]
    assert ((len(taskGraph.nodes()))==len(allocation)), "Task and allocation list length mismatch"
    for task, alloc in zip(taskGraph.nodes(), allocation):
        if task.task_type == "Sending":
            sendTask = sendTaskFactory.Create()
            sendTask.DoInitialize()
            network.taskApps.Get(alloc).AddTask(sendTask)
            task.node = list(networkGraph.nodes())[alloc]
            taskList.append(sendTask)
        elif task.task_type == "Processing":
            procTask = procTaskFactory.Create()
            procTask.DoInitialize()
            network.taskApps.Get(alloc).AddTask(procTask)
            task.node = list(networkGraph.nodes())[alloc]
            taskList.append(procTask)
        elif task.task_type == "Actuating":
            actTask = actTaskFactory.Create()
            actTask.DoInitialize()
            network.taskApps.Get(alloc).AddTask(actTask)
            task.node = list(networkGraph.nodes())[alloc]
            taskList.append(actTask)
        else:
            print(f"UNRECOGNIZED TASK TYPE {task.task_type}")
            raise exceptions.NoValidNodeException
    for nxTask, nsTask in zip (taskGraph.nodes(), taskList):
        assert nxTask.node is not None, "Nx task has no node"
        for outTask in nxTask.outputs:
            assert outTask.node is not None, "node is none in pathfinding"
            path = nx.shortest_path(networkGraph, source = nxTask.node, target = outTask.node)
            paired_task = taskList[outTask.taskId-1]
            if len(path) <= 2:
                nsTask.AddSuccessor(paired_task)
                paired_task.AddPredecessor(nsTask)
            else:
                relayList = []
                for pathNode in path:
                    relayTask = relayTaskFactory.Create()
                    relayTask.DoInitialize()
                    network.taskApps.Get(pathNode).AddTask(relayTask)
                    relayList.append(relayTask)
                for i, relayTask in enumerate(relayList):
                    if i == 0:
                        nsTask.AddSuccessor(relayList[i])
                        relayTask.AddPredecessor(nsTask)
                        relayTask.AddSuccessor(relayList[i+1])
                    elif (i < len(relayList)-1):
                        relayTask.AddSuccessor(relayList[i+1])
                        relayTask.AddPredecessor(relayList[i-1])
                    else:
                        relayTask.AddSuccessor(paired_task)
                        relayTask.AddPredecessor(relayList[i-1])
                        paired_task.AddPredecessor(relayTask)
    taskList = []
    controlTask = 0                









def evaluate(allocation, graphs):
    time = 0
    latency = 0
    #taskGraph = graphs[1]
    energy_list = graphs[2]
    networkGraph = Line(len(graphs[0].nodes()), energy=energy_list)
    #print("Building Network")
    
    network = Network(networkGraph)
    taskGraph = TwoTaskWithProcessing(networkGraph)   
    
    to_remove = []
    for node in networkGraph.nodes():
        if node.energy < 1:
            to_remove.append(node)
    networkGraph.remove_nodes_from(to_remove)
    
    if not(nx.is_connected(networkGraph)):
        return 0,99999
    createTasksFromGraph(network, taskGraph, allocation)
    #print(f"nodes left: {len(networkGraph.nodes())}")
    #print("Starting Simulation")
    latency_list = []
    energy_list = []
    time = []
    def getTime(time = []):
        time.append(ns.core.Simulator.Now().GetSeconds())    
    ns.core.Simulator.ScheduleDestroy(network.getLatency, latency_list)
    ns.core.Simulator.ScheduleDestroy(network.getEnergy, energy_list)
    ns.core.Simulator.ScheduleDestroy(getTime, time)
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()
    #print(latency_list)
    latency = max(latency_list)
    if latency == 0:
        latency = 99999
    #print(f"latency: {latency}")
    #print(energy_list)
    #energy = np.mean(energy_list)
    time = np.mean(time)
    #print(f"lifetime: {time}")
    network.cleanUp()
    network = 0
    taskGraph = 0
    networkGraph = 0
    
    return -time, latency

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
    nNodes = 20
    networkGraph = Line(nNodes)
    taskGraph = TwoTaskWithProcessing(networkGraph)
    
    nWifi = int(cmd.nWifi)
    allocation = [x for x in range(nNodes)]
    time, latency = evaluate(allocation, [networkGraph, taskGraph, [5]*nNodes]) 
    print(time)
    print(latency)
