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



class Network:
    def __init__(self, nodeCount = 10, positionSettings = {}, mobilitySettings = {}, appSettings= {}, initEnergyJ = 1000):
        self.nodeContainer = self.initNodes(nodeCount)
        self.emptyNodeContainer = ns.network.NodeContainer()
        self.mobilityHelper = self.initPositionAndMobility(positionSettings, mobilitySettings)
        self.lrWpanHelper = ns.lr_wpan.LrWpanHelper()
        self.lrWpanDeviceContainer = self.initLrWpan()
        #self.energySourceContainer = self.initEnergy(initEnergyJ)
        self.sixLowPanContainer = self.initSixLowPan()
        self.ipv6Interfaces = self.initIpv6()
        #self.apps = self.installPingApp(appSettings)
        self.apps = self.installOnOffApp(appSettings)
        
        
        #TODO: Create generic app model and install on all nodes. (Cant add apps to nodes after simstart) 
        self.currentAllocation = None
        self.enablePcap()

    def initNodes(self, nodeCount : int):
        nodeContainer = ns.network.NodeContainer()
        nodeContainer.Create(nodeCount)
        return nodeContainer 

    def initPositionAndMobility(self, posSettings, mobilitySettings):
        mobility = ns.mobility.MobilityHelper()
        posAllocator = posSettings['allocator']
        assert posAllocator == "ns3::GridPositionAllocator", f"only GridPositionAllocator supported, not {posAllocator}"
        #TODO: Add xml/json specifying which attributes need to be set for which positon allocation for checking
        minX = ns.core.DoubleValue(posSettings["MinX"])
        minY = ns.core.DoubleValue(posSettings["MinY"])
        deltaX = ns.core.DoubleValue(posSettings["DeltaX"])    
        deltaY = ns.core.DoubleValue(posSettings["DeltaY"])
        gridWidth = ns.core.UintegerValue(posSettings["GridWidth"])
        layoutType = ns.core.StringValue(posSettings["LayoutType"])
        mobility.SetPositionAllocator(posAllocator,
            "MinX", minX,
            "MinY", minY,
            "DeltaX", deltaX,
            "DeltaY", deltaY,
            "GridWidth", gridWidth,
            "LayoutType", layoutType)
        mobilityModel = mobilitySettings['model']
        assert mobilityModel == "ns3::ConstantPositionMobilityModel", f"only ConstantPositionMobilityModel supported, not {mobilityModel}"
        #TODO:Add xml,json specifying which attributes need to be set for which mobility model for checking
        mobility.SetMobilityModel(mobilityModel)
        mobility.Install(self.nodeContainer)        
        return mobility 

    def initLrWpan(self, PanID = 0):
        lrWpanDeviceContainer = self.lrWpanHelper.Install(self.nodeContainer)
        self.lrWpanHelper.AssociateToPan(lrWpanDeviceContainer, PanID)
        return lrWpanDeviceContainer 


    def initEnergy(self, initEnergyJ):
        
        liIonEnergySourceHelper = ns.energy.LiIonEnergySourceHelper()
        liIonEnergySourceHelper.Set("LiIonEnergySourceInitialEnergyJ", ns.core.DoubleValue(initEnergyJ))
        energySourceContainer = liIonEnergySourceHelper.Install(self.nodeContainer)
        
        for i in range(self.nodeContainer.GetN()):
            energyModel = ns.lr_wpan.LrWpanRadioEnergyModel()
            energyModel.AttachPhy(self.lrWpanDeviceContainer.Get(i).GetPhy())
            energyModel.SetEnergySource(energySourceContainer.Get(i))
            energySourceContainer.Get(i).AppendDeviceEnergyModel(energyModel)
        
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

    
    def installPingApp(self, appSettings):
        nWifi = self.nodeContainer.GetN()
        ping6 = ns.internet_apps.Ping6Helper()
        ping6.SetLocal(self.ipv6Interfaces.GetAddress(0,1))
        ping6.SetRemote(self.ipv6Interfaces.GetAddress(1,1))

        pSize = appSettings['packetSize']
        pCount = appSettings['packetCount']
        pInterval = appSettings['packetInterval']
        ping6.SetAttribute("PacketSize", ns.core.UintegerValue(pSize))
        ping6.SetAttribute("MaxPackets", ns.core.UintegerValue(pCount))
        ping6.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds(pInterval)))


        singlecontainer = ns.network.NodeContainer(self.nodeContainer.Get(0))
        apps = ping6.Install(singlecontainer)
        apps.Start(ns.core.Seconds(2))
        apps.Stop(ns.core.Seconds(120))
        return apps
   
    def installOnOffApp(self, appSettings):
        remoteAddress = self.ipv6Interfaces.GetLinkLocalAddress(1)
        remoteAddressWithSocket = ns.network.Inet6SocketAddress(remoteAddress, 2323)
        
        localAddress = self.ipv6Interfaces.GetLinkLocalAddress(0)
        localAddressWithSocket = ns.network.Inet6SocketAddress(localAddress, 2323)
        
        onOff = ns.applications.OnOffHelper("ns3::Ipv6RawSocketFactory", remoteAddressWithSocket)
        print(remoteAddress)

        onStream =  ns.core.ConstantRandomVariable()
        
        
        
        onStream.SetAttribute("Constant", ns.core.DoubleValue(10))
        offStream =  ns.core.ConstantRandomVariable()
        offStream.SetAttribute("Constant", ns.core.DoubleValue(10))
        
        onPtr = ns.core.PointerValue(onStream)
        offPtr = ns.core.PointerValue(offStream)
        
        onOff.SetAttribute("OnTime", onPtr)
        onOff.SetAttribute("OffTime", offPtr)
        onOff.SetAttribute("PacketSize", ns.core.UintegerValue(1000))
        onOff.SetAttribute("DataRate", ns.network.DataRateValue(ns.network.DataRate(500)))
        #onOff.SetAttribute("Local", ns.network.AddressValue(localAddressWithSocket))
        singleContainer = ns.network.NodeContainer(self.nodeContainer.Get(0))
        apps = onOff.Install(singleContainer)
        local = ns.network.AddressValue()
        remote = ns.network.AddressValue()

        apps.Get(0).GetAttribute("Local",local)
        apps.Get(0).GetAttribute("Remote",remote)
        print(local.Get())
        print(remote.Get())
        
        
        apps.Start(ns.core.Seconds(5))
        apps.Stop(ns.core.Seconds(35))
        return apps

    def enablePcap(self):
        self.lrWpanHelper.EnablePcapAll("network-wrapper", True)
        

    def deactivateNode(self, node = 0):
        #l = ns.core.ObjectPtrContainerValue()
        #self.nodeContainer.Get(0).GetAttribute("DeviceList", l)
        #for i in range(l.GetN()):
        #    print(l.Get(i))
        #    print(dir(l.Get(i)))
        
        typeID = ns.core.TypeId.LookupByName('ns3::Ipv6')
        
        self.emptyNodeContainer.Add(self.nodeContainer.Get(node))
        print(f"Added node with index {node} to empty nodes, nodeID: {self.emptyNodeContainer.Get(0).GetId()}")
        

        for i in range(self.nodeContainer.Get(node).GetObject(typeID).GetNInterfaces()):
            print(f"NetDevice: {self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).GetDevice()}")
            
            if i ==1:
                print(f"Underlying net Device: {self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).GetDevice().GetNetDevice()}")
                self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).SetDown()
                #self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).GetDevice().GetNetDevice().LinkDown()
            print(f"Interface IsUp: {self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).IsUp()}")
            print(f"Device IsLinkUp: {self.nodeContainer.Get(node).GetObject(typeID).GetInterface(i).GetDevice().IsLinkUp()}")


        

        #typeID0 = ns.core.TypeId.LookupByName('ns3::SixLowPanNetDevice')
        #print(f"sixlowpandevices: {self.nodeContainer.Get(0).GetObject(typeID1)}")
        #typeID2 = ns.core.TypeId.LookupByName('ns3::Ipv6Interface')
        #print(f"sixlowpandevices: {self.nodeContainer.Get(-1).GetObject(typeID1).GetObject(typeID2)}")
        
    

    def __str__(self):
        retVal = ""
        for i in range(self.nodeContainer.GetN()):
            mob = self.nodeContainer.Get(i).GetObject(ns.mobility.MobilityModel.GetTypeId())
            device = self.sixLowPanContainer.Get(i)
            retVal += f"Device {i}: \n {mob.GetPosition()} \n {self.ipv6Interfaces.GetAddress(i,0)} \n {self.ipv6Interfaces.GetAddress(i,1)} \n"
        return retVal 






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
    
    
    nWifi = int(cmd.nWifi)
    initEnergyJ = 1

    if verbose == "True":
        ns.core.LogComponentEnable("Ping6Application", ns.core.LOG_LEVEL_INFO)
        ns.core.LogComponentEnable("LrWpanHelper", ns.core.LOG_LEVEL_INFO) 
        ns.core.LogComponentEnable("OnOffApplication", ns.core.LOG_LEVEL_INFO)
        #ns.core.LogComponentEnable('BasicEnergySource', ns.core.LOG_LEVEL_INFO)
        #ns.core.LogComponentEnable('LrWpanRadioEnergyModel', ns.core.LOG_LEVEL_ALL)
        #ns.core.LogComponentEnable('InternetStackHelper', ns.core.LOG_ALL)
        #ns.core.LogComponentEnable('Ipv6AddressHelper', ns.core.LOG_ALL)
        #ns.core.LogComponentEnable('SixLowPanNetDevice', ns.core.LOG_LEVEL_INFO) 
        #ns.core.LogComponentEnable('LrWpanPhy', ns.core.LOG_LEVEL_ALL)
    
    posDict = {"allocator" : "ns3::GridPositionAllocator", 
                                "MinX" : ns.core.DoubleValue(0.0), 
				"MinY" : ns.core.DoubleValue (0.0), 
                                "DeltaX" : ns.core.DoubleValue(5.0), 
                                "DeltaY" : ns.core.DoubleValue(5.0), 
                                "GridWidth" : ns.core.UintegerValue(10), 
                                "LayoutType" : ns.core.StringValue("RowFirst")}
    mobDict = {"model" : "ns3::ConstantPositionMobilityModel"}
    
    appSettings = {'packetSize' : 10,
                    'packetCount' : 10,
                    'packetInterval' : 1}
    
    
    print("Building Network")
    test = Network(nodeCount= nWifi, positionSettings=posDict, mobilitySettings=mobDict, appSettings=appSettings, initEnergyJ = initEnergyJ)

    print(test)
    #ns.core.Simulator.Schedule(ns.core.Seconds(5), test.deactivateNode, 1)
    
    print("Starting Simulation")
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()








