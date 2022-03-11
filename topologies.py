import networkx as nx
import numpy.linalg as la
import numpy as np
import exceptions
import topologies
from task import Task

from itertools import combinations

class GraphNode:
    def __init__(self, posx = 0, posy = 0 , energy = 0):
        self.posx = posx
        self.posy = posy
        self.energy = energy
        self.pos = np.array([posx,posy])
        self.nTasks = 0
        self.nSending = 0
        self.nProcessing = 0
        self.nActuating= 0
        self.successors=0
        self.predecessors=0
    
    def __str__(self):
        return(f"Graphnode at: {self.pos} with energy: {self.energy}, nTasks: {self.nTasks}, nS: {self.nSending}, nP: {self.nProcessing}, nA: {self.nActuating}")

    def update_energy(self, energy):
        self.energy = energy

    def update_task_data(self,task):
        self.nTasks += 1
        if task.task_type == "Sending":
            self.nSending += 1
        elif task.task_type == "Processing":
            self.nProcessing +=1
        elif task.task_type == "Actuating":
            self.nActuating +=1
        self.successors += len(task.outputs)
        self.predecessors += len(task.inputs)
          

def Star(energy_list = [], **kwargs):
    G = nx.OrderedGraph()
    x0 = -160
    y0 = 0
    i = 0
    
    node = GraphNode(x0-40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0-40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0, y0, energy_list[i])
    G.add_node(GraphNode(x0, y0, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)

    
    x0 = 160
    y0 = 0
    node = GraphNode(x0-40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0-40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0, y0, energy_list[i])
    G.add_node(GraphNode(x0, y0, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)
    
    x0 = 0
    y0 = 0
    node = GraphNode(x0-40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0-40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0, y0, energy_list[i])
    G.add_node(GraphNode(x0, y0, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)


    x0 = 0
    y0 = 160
    node = GraphNode(x0-40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0-40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0, y0, energy_list[i])
    G.add_node(GraphNode(x0, y0, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)
    
    x0 = 0
    y0 = -160
    node = GraphNode(x0-40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0-40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0-40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0, y0, energy_list[i])
    G.add_node(GraphNode(x0, y0, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0-40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0-40, energy_list[i]), pos = node.pos, energy = node.energy)
    node = GraphNode(x0+40, y0+40, energy_list[i])
    G.add_node(GraphNode(x0+40, y0+40, energy_list[i]), pos = node.pos, energy = node.energy)
    for node1,node2 in combinations(G.nodes(),2):
        dist = la.norm(node1.pos - node2.pos)
        if dist <= 100:
            G.add_edge(node1,node2)
    return G

def Grid(dimx = 9,dimy = 9, deltax=100, deltay=100, energy_list=[], **kwargs):
    "Create a  grid network with a 4-neighborhood"
    if len(energy_list) == 0:
        print("no energy list supplied for grid creation, aborting")
        return None
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


def ManHattan(dimx = 9,dimy = 9, deltax=100, deltay=100, energy_list=[], posList = [], mobileNodeCount = 0, **kwargs):
    G= nx.OrderedGraph()
    if len(posList) == 0:
        #print("no pos list supplied for grid creation, aborting")
        #return None
        if len(energy_list) == 0:
            print("no energy list supplied for grid creation, aborting")
            return None
        x=50
        y=50
        
        #create fixed nodes:
        for i in range(dimx):
            for j in range(dimy):
              #we want to increase x with j and i with j (fill rows first)
              node = GraphNode(x+j*deltax,y+i*deltay, energy_list[j+i*dimx])
              G.add_node(GraphNode(x+j*deltax,y+i*deltay, energy_list[j+i*dimx]), pos=node.pos, energy=node.energy)
        x= 0
        y = 0
        intersections_x = []
        intersections_y = []
        for i in range(dimx):
            intersections_x.append(x+i*deltax) 
            intersections_y.append(y+i*deltax)
        np.random.seed(kwargs['seed'])
        static_nodes = dimx*dimy
        for i in range(mobileNodeCount):
            node = GraphNode(np.random.choice(intersections_x), np.random.choice(intersections_y), energy_list[static_nodes+i])
            G.add_node(GraphNode(np.random.choice(intersections_x), np.random.choice(intersections_y), energy_list[static_nodes+i]), pos=node.pos, energy = node.energy)
    else:
        for pos, energy in zip(posList,energy_list):
            node = GraphNode(pos[0],pos[1],energy)
            G.add_node(GraphNode(pos[0],pos[1],energy), pos =node.pos, energy = node.energy)
    for node1,node2 in combinations(G.nodes(),2):
        dist = la.norm(node1.pos - node2.pos)
        if dist <= 100:
            G.add_edge(node1,node2)
    return G



def Line(nNodes=100, deltax = 100, energy_list = [1000]*25, **kwargs):
    G= nx.OrderedGraph()
    for i in range(nNodes):
        node = GraphNode(i*deltax,0,energy_list[i])
        G.add_node(node, pos=node.pos, energy=node.energy)
    for node1,node2 in combinations(G.nodes(),2):
        dist = la.norm(node1.pos - node2.pos)
        if dist <= 100:
            G.add_edge(node1,node2)
    return G

def TwoTask(networkGraph = None, nTasks=0, deltax = 100, mobileNodeCount = 0,**kwargs):
    #reset global taskId counter
    assert networkGraph is not None, "Network Graph for Task Gen is None"
    #assert nTasks > 1, "Cant create TwoTaskWithProcessing with less than 2 tasks"
    Task.taskId = 1
    G = nx.OrderedDiGraph()
    pos1 = list(networkGraph.nodes())[0].pos
    bound1 = np.array([np.array([pos1[0]-deltax+1,pos1[0]+deltax-1]), np.array([pos1[1]-deltax+1,pos1[1]+deltax-1])])
    pos2 = list(networkGraph.nodes())[-(mobileNodeCount+1)].pos
    bound2 = np.array([np.array([pos2[0]-deltax+1,pos2[0]+deltax-1]), np.array([pos2[1]-deltax+1,pos2[1]+deltax-1])])
    first_constraint = {'location' : bound1}
    task1 = Task(first_constraint)
    G.add_node(task1)
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
    
def TwoTaskWithProcessing(networkGraph = None, nTasks=0, deltax = 100, **kwargs):
    #reset global taskId counter
    assert networkGraph is not None, "Network Graph for Task Gen is None"
    #assert nTasks > 1, "Cant create TwoTaskWithProcessing with less than 2 tasks"
    
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


def EncodeDecode(networkGraph = None, nTasks = 19, deltax = 100, deltay = 100, mobileNodeCount = 0, verbose = False, **kwargs):
    assert networkGraph is not None, "Network Graph for Task Gen is None"
    Task.taskId = 1
    G = nx.OrderedDiGraph()
    ndim = kwargs['dimx']
    #encoding happens on the left half of the network,  0 - int(ndim/2)
    validTaskCounts = [5, 13, 19, 25, 31, 37, 43,49,55,61,67,73,79]
    if nTasks ==5:
        n_outer = 1
        n_inner = 1
    else:
        n_outer = int((nTasks-1)/3)
        n_inner = int((n_outer)/2)
    assert nTasks in validTaskCounts, f"{nTasks} is not a valid task count for the EncodeDecode Setup! \n Valid Setups: {validTaskCounts}"
    staticNodes = list(networkGraph.nodes)[:ndim**2]
    posEncode = np.array([list(staticNodes)[0].pos[0]-deltax+1, list(staticNodes)[int(ndim/2)].pos[0]+deltax-1])
    boundEncode = np.array([posEncode, np.array([-np.inf, np.inf])])
    if verbose:
        print(f"dims : {ndim}")
        print(f"Boundary for encode: {boundEncode}")
    posDecode = np.array([list(staticNodes)[int(ndim/2)].pos[0]-deltax+1, list(staticNodes)[-1].pos[0]+deltax-1])
    boundDecode = np.array([posDecode, np.array([-np.inf, np.inf])])
    if verbose:
        print(f"Boundary for Decode: {boundDecode}")
    posCenter = np.array([list(staticNodes)[int(ndim/2)].pos[0]-deltax+1, list(staticNodes)[int(ndim/2)].pos[0]+deltax-1])
    boundCenter = np.array([posCenter, np.array([-np.inf, np.inf])])
    if verbose:
        print(f"Boundary for center: {boundCenter}")

    decode_constraint = {'location' : boundDecode}
    encode_constraint = {'location' : boundEncode}
    center_constraint = {'location' : boundCenter}
    for i in range(n_outer):
        G.add_node(Task(encode_constraint))
    for i in range(n_inner):
        G.add_node(Task())
    G.add_node(Task())
    for i in range(n_inner):
        G.add_node(Task())
    for i in range(n_outer):
        G.add_node(Task(decode_constraint))

    
    #add edges between outer left and inner left:
    for i in range(n_outer):
        for j in range(n_outer, n_outer+n_inner):
            G.add_edge(list(G.nodes())[i], list(G.nodes())[j])
    
    #edges between inner left and center
    for i in range(n_outer, n_outer+n_inner):
        G.add_edge(list(G.nodes())[i], list(G.nodes())[n_outer+n_inner])
    
    for i in range(n_outer+n_inner+1, n_outer+n_inner+1+n_inner):
        G.add_edge(list(G.nodes())[n_outer+n_inner], list(G.nodes())[i])
        for j in range(nTasks - n_outer, nTasks):
            G.add_edge(list(G.nodes())[i], list(G.nodes())[j])
        
    for task in G.nodes():
        task.set_topology(G)

    return G

     




def OneSink(networkGraph = None, deltax = 100, deltay = 100, verbose = False, sink_location = 'center', **kwargs):
    assert networkGraph is not None, "Network Graph for Task Gen is None"
    Task.taskId = 1
    G = nx.OrderedDiGraph()
    if kwargs['network_creator'] == topologies.Grid:
        ndim = kwargs['dimx']
        posCenterx = np.array([list(networkGraph.nodes())[int(ndim/2)].pos[0]-101, list(networkGraph.nodes())[int(ndim/2)].pos[0]+101])
        posCentery = np.array([list(networkGraph.nodes())[int(ndim/2)*ndim].pos[1]-101, list(networkGraph.nodes())[int(ndim/2)*ndim].pos[1]+101])
        boundCenter = np.array([posCenterx, posCentery])
    if kwargs['network_creator'] == topologies.Star:
        boundCenter = np.array([[-50, 50], [-50,50]])
        boundLeft = np.array([[-250, -50], [-50,50]])
        boundRight = np.array([[50, 250], [-50,50]])
        boundTop = np.array([[-50, 50], [50, 250]])
        boundBottom = np.array([[-50, 50], [-250, -50]])
    else:
        midTask = int(kwargs['nTasks']/2) #actually right task of the two middling tasks
        #includes the left neighbor as a possible sink task
        posCenterx = np.array([list(networkGraph.nodes())[midTask].pos[0]-101, list(networkGraph.nodes())[midTask].pos[0]+1])
        boundCenter = np.array([posCenterx, np.array([-np.inf, np.inf])])
    center_constraint = {'location' : boundCenter}
    if verbose:
        print(f"Boundary for center: {boundCenter}")
    
    
    if kwargs['network_creator'] == topologies.Star:
        for i in range(1):
            G.add_node(Task({'location' : boundCenter}))
        for i in range(2):
            G.add_node(Task({'location' : boundLeft}))
        for i in range(2):
            G.add_node(Task({'location' : boundRight}))
        for i in range(2):
            G.add_node(Task({'location' : boundTop}))
        for i in range(2):
            G.add_node(Task({'location' : boundBottom}))
    
        for i in range(8):
            G.add_edge(list(G.nodes())[i+1], list(G.nodes())[0])


    else:
        for i in range(kwargs['nTasks']-1):
            G.add_node(Task({'location' : None}))
        


        G.add_node(Task(center_constraint))

        for i in range(kwargs['nTasks']-1):
            G.add_edge(list(G.nodes())[i], list(G.nodes())[-1])

    for task in G.nodes():
        task.set_topology(G)
    return G

network_topologies = { 'Grid' : Grid,
                       'Line' : Line,
                       'Manhattan' : ManHattan,
                       'Star' : Star}

task_topologies = { 'TwoTask' : TwoTask,
                    'TwoTaskWithProcessing' : TwoTaskWithProcessing,
                    'EncodeDecode' : EncodeDecode,
                    'OneSink' : OneSink,
                    }





