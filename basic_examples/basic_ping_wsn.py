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
cmd.nWifi = 3
cmd.tracing = "True"

cmd.AddValue("nWifi", "Number of wifi STA devices")
cmd.AddValue("verbose", "Tell echo applications to log if true")
cmd.AddValue("tracing", "Enable pcap tracing")

cmd.Parse(sys.argv)

verbose = cmd.verbose
nWifi = int(cmd.nWifi)
tracing = cmd.tracing



if verbose == "True":
    ns.core.LogComponentEnable("UdpEchoClientApplication", ns.core.LOG_LEVEL_INFO)
    ns.core.LogComponentEnable("UdpEchoServerApplication", ns.core.LOG_LEVEL_INFO)
    ns.core.LogComponentEnable("Ping6Application", ns.core.LOG_LEVEL_INFO)

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
                                "DeltaX", ns.core.DoubleValue(60.0), 
                                "DeltaY", ns.core.DoubleValue(5.0), 
                                "GridWidth", ns.core.UintegerValue(nWifi), 
                                "LayoutType", ns.core.StringValue("RowFirst"))
                                 

mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobility.Install(nodes)

lrwpanhelper = ns.lr_wpan.LrWpanHelper()

lrwpandevicecontainer = lrwpanhelper.Install(nodes)

#something something fake pan association?
lrwpanhelper.AssociateToPan(lrwpandevicecontainer, 0)

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
    print(mob.GetPosition())
    print(ipv6interfaces.GetAddress(i,1))


#applications
print("Creating ping apps")
ping6 = ns.internet_apps.Ping6Helper()
ping6.SetLocal(ipv6interfaces.GetAddress(0,1))
#ping6.SetRemote(ipv6interfaces.GetAddress(1,1))
ping6.SetRemote(ns.network.Ipv6Address.GetAllNodesMulticast())

ping6.SetAttribute("PacketSize", ns.core.UintegerValue(10))
ping6.SetAttribute("MaxPackets", ns.core.UintegerValue(1))
ping6.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds(1)))

print("Instaling ping apps")

singlecontainer = ns.network.NodeContainer(nodes.Get(0))
apps = ping6.Install(singlecontainer)
apps.Start(ns.core.Seconds(2.0))
apps.Stop(ns.core.Seconds(3.0))


print("Starting Simulation")
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()








