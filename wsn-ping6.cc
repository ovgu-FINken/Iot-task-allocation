/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 aqqaasd* Copyright (c) 2014 Universita' di Firenze
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
#include <random>
#include <utility>
#include "ns3/actuating-task.h"
#include "ns3/application-container.h"
#include "ns3/control-task.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/nstime.h"
#include "ns3/object-factory.h"
#include "ns3/processing-task.h"
#include "ns3/ptr.h"
#include "ns3/send-task.h"
#include "ns3/energy-module.h"
#include "ns3/sixlowpan-module.h"
#include "ns3/lr-wpan-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/task-helper.h"
#include <chrono>
using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Ping6WsnExample");

int main (int argc, char **argv)
{
  auto start = std::chrono::system_clock::now();
  NS_LOG_INFO ("Create nodes.");
  NodeContainer nodes;
  nodes.Create (81);

  // Set seed for random numbers
  SeedManager::SetSeed (167);

  // Install mobility
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  Ptr<ListPositionAllocator> nodesPositionAlloc = CreateObject<ListPositionAllocator> ();
  for (int x =0; x<9;x++)
  {
    for (int y = 0; y<9;y++)
    {
      nodesPositionAlloc->Add (Vector (x*100.0,y*100.0,0.0));
    }

  }
  mobility.SetPositionAllocator (nodesPositionAlloc);
  mobility.Install (nodes);

  NS_LOG_INFO ("Create channels.");
  LrWpanHelper lrWpanHelper;
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

  
  // Energy
  LiIonEnergySourceHelper enH;
  enH.Set("LiIonEnergySourceInitialEnergyJ", DoubleValue(100.0));
  EnergySourceContainer enC = enH.InstallAll();
  

  NS_LOG_INFO ("Create Applications.");
  ObjectFactory procF;
  procF.SetTypeId("ns3::ProcessingTask");
  ObjectFactory sendF;
  sendF.SetTypeId("ns3::SendTask");
  ObjectFactory actF;
  actF.SetTypeId("ns3::ActuatingTask");
  ObjectFactory ctrlF;
  ctrlF.SetTypeId("ns3::ControlTask");

  int nTasks = 31;
  int nNodes = 81;
  int dimX = 9;
  std::vector<int> leftNodes;
  std::vector<int> rightNodes;
  for (int i = 0; i<dimX; ++i)
  {
    for (int j = 0; j<dimX/2; ++j)
    {
      leftNodes.push_back(i*dimX+j);
      rightNodes.push_back(i*dimX+dimX-1-j);
    }
  }
  TaskHelper taskHelper = TaskHelper(false,0,0);
  ApplicationContainer taskApps = taskHelper.Install(nodes);

  int n_allocated = 1;
  std::map<unsigned int, std::set<int>> alloc;
  alloc.insert({0, {0}});
  int n_outer = int((nTasks-1)/3);
  int n_inner = int(n_outer/2);
  
  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> distr(0, leftNodes.size() -1);
  
  std::vector<Ptr<SendTask>> sendTasks; 
  std::vector<Ptr<ProcessingTask>> procTasksL; 
  std::vector<Ptr<ProcessingTask>> procTasksC; 
  std::vector<Ptr<ProcessingTask>> procTasksR; 
  std::vector<Ptr<ActuatingTask>> actTasks; 
  Ptr<Object> o = ctrlF.Create();
  auto ctrlTask = DynamicCast<ControlTask>(o);
  alloc.insert({1,{2}});
  for (int senders = 0; senders<n_outer/2; ++senders)
  {
    sendTasks.push_back(DynamicCast<SendTask>(sendF.Create()));
    sendTasks.back()->DoInitialize();
    DynamicCast<TaskApplication>(taskApps.Get(n_allocated))->AddTask(sendTasks.back());
    alloc.insert({n_allocated++, {leftNodes.at(distr(eng))}});
  }
  for (int procers = 0; procers<n_inner; ++procers)
  {
    procTasksL.push_back(DynamicCast<ProcessingTask>(procF.Create()));
    procTasksL.back()->DoInitialize();
    DynamicCast<TaskApplication>(taskApps.Get(n_allocated))->AddTask(procTasksL.back());
    alloc.insert({n_allocated++, {leftNodes.at(distr(eng))}});
  }
  procTasksC.push_back(DynamicCast<ProcessingTask>(procF.Create()));
  procTasksC.back()->DoInitialize();
  DynamicCast<TaskApplication>(taskApps.Get(n_allocated))->AddTask(procTasksC.back());
  alloc.insert({n_allocated++, {int(nNodes/2)}});
  for (int procers = 0; procers<n_inner; ++procers)
  {
    procTasksR.push_back(DynamicCast<ProcessingTask>(procF.Create()));
    procTasksR.back()->DoInitialize();
    DynamicCast<TaskApplication>(taskApps.Get(n_allocated))->AddTask(procTasksR.back());
    alloc.insert({n_allocated++, {rightNodes.at(distr(eng))}});
  }
  for (int senders = 0; senders<n_outer/2; ++senders)
  {
    actTasks.push_back(DynamicCast<ActuatingTask>(actF.Create()));
    actTasks.back()->DoInitialize();
    DynamicCast<TaskApplication>(taskApps.Get(n_allocated))->AddTask(actTasks.back());
    alloc.insert({n_allocated++, {rightNodes.at(distr(eng))}});
  }
  //ctrlTask->SetInitialAllocation(alloc);
  for (auto& task : sendTasks)
  {
    for (auto& task2 : procTasksL)
    {
      task->AddSuccessor(task2);
      task2->AddPredecessor(task);
    }
  }
  for (auto& task : procTasksL)
  {
    for (auto& task2 : procTasksC)
    {
      task->AddSuccessor(task2);
      task2->AddPredecessor(task);
    }
  }
  for (auto& task : procTasksC)
  {
    for (auto& task2 : procTasksR)
    {
      task->AddSuccessor(task2);
      task2->AddPredecessor(task);
    }
  }
  for (auto& task : procTasksR)
  {
    for (auto& task2 : actTasks)
    {
      task->AddSuccessor(task2);
      task2->AddPredecessor(task);
    }
  }
  

  NS_LOG_INFO ("Run Simulation.");
  


  Simulator::Run ();
  Simulator::Stop(Time(Seconds(120)));
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
  auto end = std::chrono::system_clock::now();
  auto elapsed = end-start;
  std::cout << elapsed.count() << std::endl;
}

