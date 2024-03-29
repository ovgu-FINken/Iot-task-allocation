/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universita' di Firenze
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

// Network topology
//
//       n0    n1
//       |     |
//       =================
//        WSN (802.15.4)
//
// - ICMPv6 echo request flows from n0 to n1 and back with ICMPv6 echo reply
// - DropTail queues 
// - Tracing of queues and packet receptions to file "wsn-ping6.tr"
//
// This example is based on the "ping6.cc" example.

#include <fstream>
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/sixlowpan-module.h"
#include "ns3/lr-wpan-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/sixlowpan-net-device.h"
#include "ns3/type-id.h"




using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Ping6WsnExample");

int main (int argc, char **argv)
{
  bool verbose = false;
  uint16_t nWifi = 2;
  CommandLine cmd (__FILE__);
  cmd.AddValue ("verbose", "turn on log components", verbose);
  cmd.AddValue ("nWifi", "Number of wifi nodes", nWifi);
  cmd.Parse (argc, argv);

  if (verbose)
    {
      LogComponentEnable ("Ping6WsnExample", LOG_LEVEL_INFO);
      //LogComponentEnable ("Ipv6EndPointDemux", LOG_LEVEL_ALL);
      //LogComponentEnable ("Ipv6L3Protocol", LOG_LEVEL_ALL);
      //LogComponentEnable ("Ipv6StaticRouting", LOG_LEVEL_ALL);
      //LogComponentEnable ("Ipv6ListRouting", LOG_LEVEL_ALL);
      //LogComponentEnable ("Ipv6Interface", LOG_LEVEL_ALL);
      //LogComponentEnable ("Icmpv6L4Protocol", LOG_LEVEL_ALL);
      LogComponentEnable ("Ping6Application", LOG_LEVEL_ALL);
      //LogComponentEnable ("NdiscCache", LOG_LEVEL_ALL);
      //LogComponentEnable ("SixLowPanNetDevice", LOG_LEVEL_ALL);
    }

  NS_LOG_INFO ("Create nodes.");
  NodeContainer nodes;
  nodes.Create (nWifi);

  // Set seed for random numbers
  SeedManager::SetSeed (167);

  // Install mobility
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  mobility.SetPositionAllocator ("ns3::GridPositionAllocator", 
                                "MinX", DoubleValue (0.0), 
                				"MinY", DoubleValue (0.0), 
                                "DeltaX", DoubleValue (50.0), 
                                "DeltaY", DoubleValue (5.0), 
                                "GridWidth", UintegerValue (nWifi), 
                                "LayoutType", StringValue ("RowFirst"));
  
  mobility.Install (nodes);

  NS_LOG_INFO ("Create channels.");
  LrWpanHelper lrWpanHelper;
  // Add and install the LrWpanNetDevice for each node
  // lrWpanHelper.EnableLogComponents();
  NetDeviceContainer devContainer = lrWpanHelper.Install(nodes);
  lrWpanHelper.AssociateToPan (devContainer, 10);

  std::cout << "Created " << devContainer.GetN() << " devices" << std::endl;
  std::cout << "There are " << nodes.GetN() << " nodes" << std::endl;

  /* Install IPv4/IPv6 stack */
  NS_LOG_INFO ("Install Internet stack.");
  InternetStackHelper internetv6;
  internetv6.SetIpv4StackInstall (false);
  internetv6.Install (nodes);

  // Install 6LowPan layer
  NS_LOG_INFO ("Install 6LoWPAN.");
  SixLowPanHelper sixlowpan;
  NetDeviceContainer six1 = sixlowpan.Install (devContainer);

  NS_LOG_INFO ("Assign addresses.");
  Ipv6AddressHelper ipv6;
  ipv6.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i = ipv6.Assign (six1);

  NS_LOG_INFO ("Create Applications.");

  /* Create a Ping6 application to send ICMPv6 echo request from node zero to
   * all-nodes (ff02::1).
   */
  uint32_t packetSize = 10;
  uint32_t maxPacketCount = 5;
  Time interPacketInterval = Seconds (1.);
  Ping6Helper ping6;

  ping6.SetLocal (i.GetAddress (0, 1));
  ping6.SetRemote (i.GetAddress (nWifi-1, 1));
  // ping6.SetRemote (Ipv6Address::GetAllNodesMulticast ());

  ping6.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
  ping6.SetAttribute ("Interval", TimeValue (interPacketInterval));
  ping6.SetAttribute ("PacketSize", UintegerValue (packetSize));
  ApplicationContainer apps = ping6.Install (nodes.Get (0));
  //apps.Start (Seconds (2.0));
  //apps.Stop (Seconds (10.0));

  AsciiTraceHelper ascii;
  lrWpanHelper.EnableAsciiAll (ascii.CreateFileStream ("ping6wsn.tr"));
  lrWpanHelper.EnablePcapAll (std::string ("ping6wsn"), true);

  
  ObjectFactory taskFactory;
  taskFactory.SetTypeId("ns3::SendTask");
  Ptr<Object> t1;
  t1 = taskFactory.Create();
  //t1->Execute();



  //tasks.push_back(TaskFactory::Create("SEND", {'a', 100, 1234}));
  /**
  for (auto it = tasks.begin(); it != tasks.end(); ++it){
        it->get()->Execute();
  }
  **/
 
  TaskHelper taskHelper;
  ApplicationContainer taskApps = taskHelper.Install (nodes);
  std::cout << "ping6 " << apps.Get(0)->GetTypeId() << std::endl;
  std::cout << "task " << taskApps.Get(0)->GetTypeId() << std::endl;
  
  Ptr<Node> n = nodes.Get(0);
  Ptr<SixLowPanNetDevice> six = n->GetObject<SixLowPanNetDevice>();
  std::cout << six << std::endl;
  //taskApps.Get(0).AddTask(t1);



  //for (auto it = TaskApps.Begin(), it != taskApps.End(), it++)

  //taskApps.Get(0)->MigrateTask(0, taskApps.Get(1));
  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
}

