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
import sys

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
cmd.nWifi = 8
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
                                "DeltaX", ns.core.DoubleValue(1.0), 
                                "DeltaY", ns.core.DoubleValue(1.0), 
                                "GridWidth", ns.core.UintegerValue(nWifi), 
                                "LayoutType", ns.core.StringValue("RowFirst"))
                                

mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobility.Install(nodes)

#print(dir(ns.core))

lrwpanhelper = ns.lr_wpan.LrWpanHelper()

lrwpandevicecontainer = lrwpanhelper.Install(nodes)

#Associate devices to specific PAN
lrwpanhelper.AssociateToPan(lrwpandevicecontainer, 0)

#Energy Model
basicEnergySourceHelper = ns.energy.BasicEnergySourceHelper()
basicEnergySourceHelper.Set("BasicEnergySourceInitialEnergyJ", ns.core.DoubleValue(100))

testNode=nodes.Get(0)
energySources = basicEnergySourceHelper.Install(nodes)
testDevice=lrwpandevicecontainer.Get(0)
testEnergySource = energySources.Get(0)

energymodel = ns.lr_wpan.LrWpanRadioEnergyModel()
energymodel.AttachPhy(testDevice.GetPhy())
energymodel.SetEnergySource(testEnergySource)
testEnergySource.AppendDeviceEnergyModel(energymodel)


#stack install
ipv6stack = ns.internet.InternetStackHelper()
ipv6stack.SetIpv4StackInstall(False)
ipv6stack.Install(nodes)


#6lowpan
sixlowpan = ns.sixlowpan.SixLowPanHelper()
sixlowpancontainer = sixlowpan.Install(lrwpandevicecontainer)



ipv6address = ns.internet.Ipv6AddressHelper()
ipv6address.SetBase(ns.network.Ipv6Address("2001:1::"), ns.network.Ipv6Prefix(64))
ipv6interfaces = ipv6address.Assign(sixlowpancontainer)





for i in range(nodes.GetN()):
    mob = nodes.Get(i).GetObject(ns.mobility.MobilityModel.GetTypeId())
    device = sixlowpancontainer.Get(i)
    #set mesh-under protocol
    #device.SetAttribute("UseMeshUnder", ns.core.BooleanValue(True))
    #device.SetAttribute("MeshUnderRadius", ns.core.UintegerValue(10))
    #device.SetAttribute("MeshCacheLength", ns.core.UintegerValue(50))
    #print some stuff
    print(f"Device {i}:")
    #print(mob.GetPosition())
    print(f"Sixlowpan address: {device.GetAddress()}")
    print(f"link local: {ipv6interfaces.GetAddress(i,0)}")
    print(f"whatever this is: {ipv6interfaces.GetAddress(i,1)}")
    print()



taskFactory = ns.core.ObjectFactory()
taskFactory.SetTypeId("ns3::ProcessingTask")
taskFactory.Set ("StartTime", ns.core.TimeValue(ns.core.Seconds(1))) 
taskFactory.Set ("StopTime", ns.core.TimeValue(ns.core.Seconds(12))) 
taskFactory.Set ("Port", ns.core.UintegerValue(1337))

sendTaskFactory = ns.core.ObjectFactory()
sendTaskFactory.SetTypeId("ns3::SendTask")
sendTaskFactory.Set ("StartTime", ns.core.TimeValue(ns.core.Seconds(1))) 
sendTaskFactory.Set ("Interval", ns.core.TimeValue(ns.core.Seconds(3)))
taskFactory.Set ("Port", ns.core.UintegerValue(4114))

controlFactory = ns.core.ObjectFactory()
controlFactory.SetTypeId("ns3::ControlTask")
c1 = controlFactory.Create()


taskHelper = ns.applications.TaskHelper()
taskApps = taskHelper.Install(nodes)

alloc = [(0,[0]),
	 (1,[0]),
	 (2,[0]),
	 (3,[1]),
	 (4,[1]),
	 (5,[2]),
	 (6,[2]),
	 (7,[3]),
	 (8,[3]),
	 (9,[4]),
	 (10,[4]),
	 (11,[5]),
	 (12,[5]),
	 (13,[6]),
	 (14,[6]),
	 (15,[7]),
	 (16,[7]),
	 ]

c1.SetInitialAllocation(alloc)
c1.DoInitialize()
taskApps.Get(0).AddTask(c1)

for i in range(taskApps.GetN()):
    #print(f"TaskApp {i}:")
    #print(taskApps.Get(i).GetTypeId())
    print("proctask")
    t1 = taskFactory.Create()
    print("sendtask")
    t2 = sendTaskFactory.Create()
    #we need to manually initialize these because we do not aggregate them 
    t1.DoInitialize()
    t2.DoInitialize()
    t2.AddSuccessor(t1)
    t1.AddPredecessor(t2)
    taskApps.Get(i).AddTask(t1)
    taskApps.Get(i).AddTask(t2)
    print()

alloc = [(0,[0]),
	 (1,[7]),
	 (2,[7]),
	 (3,[6]),
	 (4,[6]),
	 (5,[5]),
	 (6,[5]),
	 (7,[4]),
	 (8,[4]),
	 (9,[3]),
	 (10,[3]),
	 (11,[2]),
	 (12,[2]),
	 (13,[1]),
	 (14,[1]),
	 (15,[1]),
	 (16,[1]),
	 ]


ascii = ns.network.AsciiTraceHelper()
lrwpanhelper.EnablePcapAll("basic-wsn-example", True)

ns.core.Simulator.Schedule(ns.core.Seconds(10), c1.Reallocate, alloc, taskApps)

print("Starting Simulation")
ns.core.Simulator.Stop(ns.core.Seconds(20))
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()







