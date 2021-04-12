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
cmd.nWifi = 2
cmd.tracing = "True"
cmd.distance = 50

cmd.AddValue("nWifi", "Number of wifi STA devices")
cmd.AddValue("verbose", "Tell echo applications to log if true")
cmd.AddValue("tracing", "Enable pcap tracing")

cmd.AddValue("distance", "Grid x-distance")

cmd.Parse(sys.argv)

verbose = cmd.verbose
nWifi = int(cmd.nWifi)
tracing = cmd.tracing
distance = int(cmd.distance)


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
                                "DeltaX", ns.core.DoubleValue(distance), 
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



#energyModelContainer = ns.energy.DeviceEnergyModelContainer()
#for i in range(lrwpandevicecontainer.GetN()):
#    testDevice=lrwpandevicecontainer.Get(i)
#    testEnergySource = energySources.Get(i)
#
#    energymodel = ns.lr_wpan.LrWpanRadioEnergyModel()
#    energymodel.AttachPhy(testDevice.GetPhy())
#    energymodel.SetEnergySource(testEnergySource)
#    testEnergySource.AppendDeviceEnergyModel(energymodel)
#    energyModelContainer.Add(energymodel)


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


ping6helper = ns.internet_apps.Ping6Helper()
ping6helper.SetLocal(ipv6interfaces.GetAddress(0,1))
ping6helper.SetRemote(ipv6interfaces.GetAddress(1,1))
ping6helper.SetAttribute("PacketSize", ns.core.UintegerValue(100))
ping6helper.SetAttribute("MaxPackets", ns.core.UintegerValue(5))
ping6helper.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds(2)))
singlecontainer = ns.network.NodeContainer(nodes.Get(0))
ping6helper.Install(singlecontainer)


ascii = ns.network.AsciiTraceHelper()
lrwpanhelper.EnablePcapAll("basic-wsn-example", True)


print("Starting Simulation")
ns.core.Simulator.Stop(ns.core.Seconds(12))
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()
