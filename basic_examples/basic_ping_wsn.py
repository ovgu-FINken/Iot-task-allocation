import ns.core
import ns.visualizer
import ns.network
#import ns.netanim
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
import sys
import random
import time
# // Default Network Topology
# //
# //   Wifi 10.1.3.0
# //                 AP
# //  *    *    *    *
# //  |    |    |    |
# // n5   n6   n7   n0
# //                 
# //                
# //               

cmd = ns.core.CommandLine()
cmd.verbose = "True"
cmd.nWifi = 25
cmd.tracing = "True"

cmd.AddValue("nWifi", "Number of wifi STA devices")
cmd.AddValue("verbose", "Tell echo applications to log if true")
cmd.AddValue("tracing", "Enable pcap tracing")

cmd.Parse(sys.argv)

verbose = cmd.verbose
nWifi = int(cmd.nWifi)
tracing = cmd.tracing


if verbose == "True":
    ns.core.LogComponentEnable("Ping6Application", ns.core.LOG_LEVEL_INFO)
    ns.core.LogComponentEnable("LrWpanHelper", ns.core.LOG_LEVEL_INFO) 
    #ns.core.LogComponentEnable('BasicEnergySource', ns.core.LOG_LEVEL_ALL)
    #ns.core.LogComponentEnable('LrWpanRadioEnergyModel', ns.core.LOG_LEVEL_ALL)
    #ns.core.LogComponentEnable('InternetStackHelper', ns.core.LOG_ALL)
    #ns.core.LogComponentEnable('Ipv6AddressHelper', ns.core.LOG_ALL)
    #ns.core.LogComponentEnable('SixLowPanNetDevice', ns.core.LOG_LEVEL_INFO) 
    #ns.core.LogComponentEnable('LrWpanPhy', ns.core.LOG_LEVEL_ALL)
nodes =  ns.network.NodeContainer()
nodes.Create(nWifi)
#wifiApNode = wifiStaNodes.Get(1)

#channel = ns.wifi.YansWifiChannelHelper.Default()
#phy = ns.wifi.YansWifiPhyHelper.Default()
#phy.SetChannel(channel.Create())


#apDevices = lrpan.Install(wifiApNode)

mobility = ns.mobility.MobilityHelper()
mobility.SetPositionAllocator ("ns3::GridPositionAllocator", 
                                "MinX", ns.core.DoubleValue(0.0), 
				"MinY", ns.core.DoubleValue (0.0), 
                                "DeltaX", ns.core.DoubleValue(10.0), 
                                "DeltaY", ns.core.DoubleValue(10.0), 
                                "GridWidth", ns.core.UintegerValue(5), 
                                "LayoutType", ns.core.StringValue("RowFirst"))
                                

mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobility.Install(nodes)

#print(dir(ns.core))

lrwpanhelper = ns.lr_wpan.LrWpanHelper()

lrwpandevicecontainer = lrwpanhelper.Install(nodes)

for i in range(lrwpandevicecontainer.GetN()):
    lrwpandevicecontainer.Get(i).GetCsmaCa().SetMacMaxBE(255)
    #lrwpandevicecontainer.Get(i).GetCsmaCa().SetMacMinBE(5)
    lrwpandevicecontainer.Get(i).GetCsmaCa().SetUnitBackoffPeriod(255)
    print(lrwpandevicecontainer.Get(i).GetCsmaCa().IsUnSlottedCsmaCa())
    print(lrwpandevicecontainer.Get(i).GetMac().GetAssociationStatus())
#Associate devices to specific PAN
lrwpanhelper.AssociateToPan(lrwpandevicecontainer, 0)

#Energy Model
liIonEnergySourceHelper = ns.energy.LiIonEnergySourceHelper()
liIonEnergySourceHelper.Set("LiIonEnergySourceInitialEnergyJ", ns.core.DoubleValue(2))

energySources = liIonEnergySourceHelper.Install(nodes)



energyModelContainer = ns.energy.DeviceEnergyModelContainer()
for i in range(lrwpandevicecontainer.GetN()):
    testDevice=lrwpandevicecontainer.Get(i)
    testEnergySource = energySources.Get(i)

    energymodel = ns.lr_wpan.LrWpanRadioEnergyModel()
    energymodel.AttachPhy(testDevice.GetPhy())
    energymodel.SetEnergySource(testEnergySource)
    testEnergySource.AppendDeviceEnergyModel(energymodel)
    energyModelContainer.Add(energymodel)


#stack install
ipv6stack = ns.internet.InternetStackHelper()
ipv6staticrouting = ns.internet.Ipv6StaticRoutingHelper()
ipv6stack.SetRoutingHelper(ipv6staticrouting)
ipv6stack.SetIpv4StackInstall(False)
ipv6stack.Install(nodes)

#6lowpan
sixlowpan = ns.sixlowpan.SixLowPanHelper()
sixlowpancontainer = sixlowpan.Install(lrwpandevicecontainer)



ipv6address = ns.internet.Ipv6AddressHelper()
ipv6address.SetBase(ns.network.Ipv6Address("2001:1::"), ns.network.Ipv6Prefix(64))
ipv6interfaces = ipv6address.Assign(sixlowpancontainer)
#ipv6interfaces.SetDefaultRouteInAllNodes(0)



for i in range(lrwpandevicecontainer.GetN()):
    mac = lrwpandevicecontainer.Get(i).GetMac()
    mac.SetMacMaxFrameRetries(0)

for i in range(ipv6interfaces.GetN()):
    ipv6interfaces.SetForwarding(i,True)



for i in range(nodes.GetN()):
    mob = nodes.Get(i).GetObject(ns.mobility.MobilityModel.GetTypeId())
    icmpv6 = nodes.Get(i).GetObject(ns.internet.Icmpv6L4Protocol.GetTypeId())
    icmpv6.SetAttribute("RetransmissionTime", ns.core.TimeValue(ns.core.Seconds(15)))
    icmpv6.SetAttribute("DelayFirstProbe", ns.core.TimeValue(ns.core.Seconds(15)))
    icmpv6.SetAttribute("MaxMulticastSolicit", ns.core.IntegerValue(0))
    icmpv6.SetAttribute("MaxUnicastSolicit", ns.core.IntegerValue(0))
    icmpv6.SetAttribute("DAD", ns.core.BooleanValue(False))

    ipv6L3 = nodes.Get(i).GetObject(ns.internet.Ipv6L3Protocol.GetTypeId())
    ipv6L3.SetAttribute("SendIcmpv6Redirect", ns.core.BooleanValue(False))
    ipv6L3.SetAttribute("IpForward", ns.core.BooleanValue(False))
    
    



    device = sixlowpancontainer.Get(i)
    #set mesh-under protocol
    device.SetAttribute("UseMeshUnder", ns.core.BooleanValue(False))
    device.SetAttribute("MeshUnderRadius", ns.core.UintegerValue(3))
    #device.SetAttribute("MeshCacheLength", ns.core.UintegerValue(50))
    #print some stuff
    print(f"Device {i}:")
    #print(mob.GetPosition())
    print(f"Sixlowpan address: {device.GetAddress()}")
    print(f"link local: {ipv6interfaces.GetAddress(i,0)}")
    print(f"whatever this is: {ipv6interfaces.GetAddress(i,1)}")
    print()



procTaskFactory = ns.core.ObjectFactory()
procTaskFactory.SetTypeId("ns3::ProcessingTask")
procTaskFactory.Set ("StartTime", ns.core.TimeValue(ns.core.Seconds(1))) 

sendTaskFactory = ns.core.ObjectFactory()
sendTaskFactory.SetTypeId("ns3::SendTask")
sendTaskFactory.Set ("StartTime", ns.core.TimeValue(ns.core.Seconds(1))) 
sendTaskFactory.Set ("Interval", ns.core.TimeValue(ns.core.Seconds(3)))

actTaskFactory = ns.core.ObjectFactory()
actTaskFactory.SetTypeId("ns3::ActuatingTask")
actTaskFactory.Set ("StartTime", ns.core.TimeValue(ns.core.Seconds(1))) 

relayTaskFactory = ns.core.ObjectFactory()
relayTaskFactory.SetTypeId("ns3::RelayTask")
relayTaskFactory.Set ("StartTime", ns.core.TimeValue(ns.core.Seconds(1))) 

controlFactory = ns.core.ObjectFactory()
controlFactory.SetTypeId("ns3::ControlTask")
c1 = controlFactory.Create()


taskHelper = ns.applications.TaskHelper()
taskApps = taskHelper.Install(nodes)



#for i in range(taskApps.GetN()):
#    #print(f"TaskApp {i}:")
#    #print(taskApps.Get(i).GetTypeId())
#    print("proctask")
#    t1 = taskFactory.Create()
#    print("sendtask")
#    t2 = sendTaskFactory.Create()
#    #we need to manually initialize these because we do not aggregate them 
#    t1.DoInitialize()
#    t2.DoInitialize()
#    t2.AddSuccessor(t1)
#    t1.AddPredecessor(t2)
#    taskApps.Get(i).AddTask(t1)
#    taskApps.Get(i).AddTask(t2)
#    print()



actTask = actTaskFactory.Create()
actTask.DoInitialize()
taskApps.Get(4).AddTask(actTask)

relayTask = relayTaskFactory.Create()
relayTask.DoInitialize()
taskApps.Get(3).AddTask(relayTask)

procTask = procTaskFactory.Create()
procTask.DoInitialize()
procTask.AddSuccessor(actTask)
procTask.AddPredecessor(relayTask)
relayTask.AddSuccessor(procTask)
taskApps.Get(5).AddTask(procTask)
actTask.AddPredecessor(procTask)


for i in range(15):
    rand = (random.random())*50
    sendTaskFactory.Set ("Interval", ns.core.TimeValue(ns.core.Seconds(rand)))
    sendTask=sendTaskFactory.Create()
    print(f"Created sendtask with ID {sendTask.GetTaskId()} on node {taskApps.Get(i).GetNode().GetId()}")
    sendTask.DoInitialize()
    sendTask.AddSuccessor(relayTask)
    relayTask.AddPredecessor(sendTask)
    taskApps.Get(i).AddTask(sendTask)

relayTask = 0
#for i in range(5):
#    rand = (random.random()+random.random()+random.random()+random.random())*10
#    sendTaskFactory.Set ("Interval", ns.core.TimeValue(ns.core.Seconds(rand)))
#    sendTask=sendTaskFactory.Create()
#    print(f"Created sendtask with ID {sendTask.GetTaskId()} on node {taskApps.Get(i).GetNode().GetId()}")
#    sendTask.DoInitialize()
#    sendTask.AddSuccessor(actTask)
#    actTask.AddPredecessor(sendTask)
#    taskApps.Get(i).AddTask(sendTask)
#for i in range(3):
#    rand = (random.random()+random.random()+random.random()+random.random())*10
#    sendTaskFactory.Set ("Interval", ns.core.TimeValue(ns.core.Seconds(rand)))
#    sendTask=sendTaskFactory.Create()
#    print(f"Created sendtask with ID {sendTask.GetTaskId()} on node {taskApps.Get(i).GetNode().GetId()}")
#    sendTask.DoInitialize()
#    sendTask.AddSuccessor(actTask)
#    actTask.AddPredecessor(sendTask)
#    taskApps.Get(i).AddTask(sendTask)
sendTask = 0

#sendtask=sendTaskFactory.Create()
#procTask=taskFactory.Create()
#actTask = actTaskFactory.Create()
#relayTask = relayTaskFactory.Create()
#
alloc = [(0,[0]),
	 (1,[0]),
	 (2,[2]),
	 (3,[3]),
	 ]

c1.SetInitialAllocation(alloc)
c1.DoInitialize()
taskApps.Get(0).AddTask(c1)

#
#sendtask.DoInitialize()
#procTask.DoInitialize()
#actTask.DoInitialize()
#relayTask.DoInitialize()
#
#sendtask.AddSuccessor(relayTask)
#relayTask.AddPredecessor(sendtask)
#relayTask.AddSuccessor(procTask)
#procTask.AddPredecessor(relayTask)
#procTask.AddSuccessor(actTask)
#actTask.AddPredecessor(procTask)
#
#taskApps.Get(0).AddTask(sendtask)
#taskApps.Get(1).AddTask(relayTask)
#taskApps.Get(2).AddTask(procTask)
#taskApps.Get(3).AddTask(actTask)
#
#sendtask = 0
#procTask = 0
#relayTask = 0
#actTask=0

alloc = [(0,[3]),
	 (1,[2]),
	 (2,[1]),
	 (3,[0]),
	 ]

#alloc = [(0,[0]),
#	 (1,[0]),
#	 (2,[7]),
#	 (3,[1]),
#	 (4,[6]),
#	 (5,[2]),
#	 (6,[5]),
#	 (7,[3]),
#	 (8,[4]),
#	 (9,[4]),
#	 (10,[3]),
#	 (11,[5]),
#	 (12,[2]),
#	 (13,[6]),
#	 (14,[1]),
#	 (15,[7]),
#	 (16,[1]),
#	 ]

#ascii = ns.core.OfStreamWrapper("wifi-ap.tr") # create the file:
#lrwpanhelper.EnablePcapAll("basic-wsn-example", True)
#lrwpanhelper.EnableAsciiAll (ascii)



#ns.core.Simulator.Schedule(ns.core.Seconds(5), c1.Reallocate, alloc, taskApps)


def printlatency():
    print(actTask.GetAverageLatency()/1000000000)
    return None

def printEnergy():
    for i in range(energySources.GetN()):
        print(energySources.Get(i).GetRemainingEnergy())
    return None

def printTime():
    print(ns.core.Simulator.Now().GetSeconds())

#    mob = nodes.Get(i).GetObject(ns.mobility.MobilityModel.GetTypeId())

alloc = [(0,[0]),
	 (1,[0]),
	 (2,[1]),
	 (3,[2]),
	 ]
#ns.core.Simulator.Schedule(ns.core.Seconds(8), c1.Reallocate, alloc, taskApps)
ns.core.Simulator.ScheduleDestroy(printEnergy)
ns.core.Simulator.ScheduleDestroy(printlatency)
ns.core.Simulator.ScheduleDestroy(printTime)
start = time.time()
print("Starting Simulation")
#ns.core.Simulator.Stop(ns.core.Seconds(20))
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()
#ascii.close()
print(f"Runtime: {time.time() - start}")





