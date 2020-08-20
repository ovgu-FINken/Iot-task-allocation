import networkx as nx
import numpy.linalg as la
import numpy as np
import exceptions

from task import Task

from itertools import combinations

class GraphNode:
    def __init__(self, posx = 0, posy = 0 , energy = 0):
        self.posx = posx
        self.posy = posy
        self.energy = energy
        self.pos = np.array([posx,posy])
    
    def __str__(self):
        return(f"Graphnode at: {self.pos} with energy: {self.energy}")

def Grid(dimx = 9,dimy = 9, deltax=100, deltay=100, energy_list=[1000]*25, **kwargs):
    "Create a  grid network with a 4-neighborhood"
    x=0
    y=0
    G=nx.OrderedGraph()
    for i in range(dimx):
        for j in range(dimy):
          #we want to increase x with j and i with j (fill rows first)
          node = GraphNode(x+j*deltax,y+i*deltay, energy_list[j+i*dimx])
          G.add_node(GraphNode(x+j*deltax,y+i*deltay, energy_list[j+i*dimx]), pos=node.pos, energy=node.energy)
    for node1,node2 in combinations(G.nodes(),2):
        dist = la.norm(node1.pos - node2.pos)
        if dist <= 100:
            G.add_edge(node1,node2)
    return G

def Line(nNodes = 25, deltax = 100, energy_list = [1000]*25, **kwargs):
    G= nx.OrderedGraph()
    for i in range(nNodes):
        node = GraphNode(i*deltax,0,energy_list[i])
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
    
def TwoTaskWithProcessing(networkGraph = None, nTasks=0, deltax = 100, **kwargs):
    #reset global taskId counter
    assert networkGraph is not None, "Network Graph for Task Gen is None"
    if nTasks == 0:
        nTasks = len(networkGraph.nodes())
    Task.taskId = 1
    G = nx.OrderedDiGraph()
    pos1 = list(networkGraph.nodes())[0].pos
    bound1 = np.array([np.array([pos1[0]-1,pos1[0]+1]), np.array([pos1[1]-1,pos1[1]+1])])
    pos2 = list(networkGraph.nodes())[-1].pos
    bound2 = np.array([np.array([pos2[0]-1,pos2[0]+1]), np.array([pos2[1]-1,pos2[1]+1])])
    first_constraint = {'location' : bound1}
    task1 = Task(first_constraint)
    G.add_node(task1)
    for i in range(nTasks-2):
        G.add_node(Task())
    second_constraint = {'location' : bound2}
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


def EncodeDecode(networkGraph = None, deltax = 100, deltay = 100, verbose = False, **kwargs):
    assert networkGraph is not None, "Network Graph for Task Gen is None"
    Task.taskId = 1
    G = nx.OrderedDiGraph()
    ndim = kwargs['dimx']
    #encoding happens on the left half of the network,  0 - int(ndim/2) 
    posEncode = np.array([list(networkGraph.nodes())[0].pos[0]-1, list(networkGraph.nodes())[int(ndim/2)].pos[0]+1])
    boundEncode = np.array([posEncode, np.array([-np.inf, np.inf])])
    if verbose:
        print(f"Boundary for encode: {boundEncode}")
    posDecode = np.array([list(networkGraph.nodes())[int(ndim/2)].pos[0]-1, list(networkGraph.nodes())[-1].pos[0]+1])
    boundDecode = np.array([posDecode, np.array([-np.inf, np.inf])])
    if verbose:
        print(f"Boundary for Decode: {boundDecode}")
    posCenter = np.array([list(networkGraph.nodes())[int(ndim/2)].pos[0]-1, list(networkGraph.nodes())[int(ndim/2)].pos[0]+1])
    boundCenter = np.array([posCenter, np.array([-np.inf, np.inf])])
    if verbose:
        print(f"Boundary for center: {boundCenter}")

    decode_constraint = {'location' : boundDecode}
    encode_constraint = {'location' : boundEncode}
    center_constraint = {'location' : boundCenter}
    for i in range(6):
        G.add_node(Task(encode_constraint))
    for i in range(3):
        G.add_node(Task())
    G.add_node(Task())
    for i in range(3):
        G.add_node(Task())
    for i in range(6):
        G.add_node(Task(decode_constraint))

    

    for i in range(6):
        for j in [6,7,8]:
            G.add_edge(list(G.nodes())[i], list(G.nodes())[j])

    for i in [6,7,8]:
        G.add_edge(list(G.nodes())[i], list(G.nodes())[9])
    
    for i in [10,11,12]:
        G.add_edge(list(G.nodes())[9], list(G.nodes())[i])
        for j in [13,14,15,16,17,18]:
            G.add_edge(list(G.nodes())[i], list(G.nodes())[j])
        
    for task in G.nodes():
        task.set_topology(G)

    return G

     




def OneSink(networkGraph = None, deltax = 100, deltay = 100, verbose = False, sink_location = 'center', **kwargs):
    assert networkGraph is not None, "Network Graph for Task Gen is None"
    Task.taskId = 1
    G = nx.OrderedDiGraph()
    ndim = kwargs['dimx']
    posCenter = np.array([list(networkGraph.nodes())[int(ndim/2)].pos[0]-1, list(networkGraph.nodes())[int(ndim/2)].pos[0]+1])
    boundCenter = np.array([posCenter, np.array([-np.inf, np.inf])])
    if verbose:
        print(f"Boundary for center: {boundCenter}")







