import networkx as nx
import numpy as np
#import config
from exceptions import *

class Task():
    taskId = 1
    """Class representing a task to assign to the network"""
    def __init__(self, constraints = {}, inputs=[], outputs=[]):
        self.constraints = constraints 
        self.inputs = inputs
        self.outputs = outputs
        self.task_type = ""
        self.taskId = Task.taskId
        self.node = None
        Task.taskId += 1

    def get_constrained_nodes(self, networkGraph):
        """ Checks for nodes in the network satisfying the task constraints and returns a list of them"""
        constrained_nodes = networkGraph.nodes()
        for constraint, value in self.constraints.items():
            if constraint =='location':
                if self.constraints['location'] is not None:
                    #boundary is not the global search space, check for fitting nodes
                    constrained_nodes = [node for node in constrained_nodes if self.check_location_constraint(node.pos)]
                    if len(constrained_nodes) == 0:
                        #no node satisfying constraint, RIP
                        raise NoValidNodeException("No node satisfying positional constraints")
            else:
                #found a constraint thas is not yet implemented?!
                raise NotImplementedError(f"Constraint '{constraint}' is not implemented for constraint checking")
        return constrained_nodes

    def set_topology(self, dacg: nx.DiGraph):
        self.outputs = list(dacg.successors(self))
        self.inputs = list(dacg.predecessors(self))
        #keep bounds only where reasonable, e.g. sensing and actuating
        if len(self.outputs) >0 and len(self.inputs) >0:
            self.constraints.update({'location' : None})
        #set consumption according to task
        if len(self.inputs) == 0:
            #sensing task, expensive
            #self.properties.update({'consumption' : config.sensing_cost})
            self.task_type = "Sending"
        elif len(self.outputs) == 0:
            #actuating task, also expensive
            #self.properties.update({'consumption' : config.actuating_cost})
            self.task_type = "Actuating"
        else:
            #merely processing, low cost
            #self.properties.update({'consumption' : config.processing_cost})a
            self.task_type = "Processing"
    
    def check_bounds(self, node):
        """ Check if the task can be fulfilled by the specified node """
        if self.constraints['location'] is None:
            return True
        if self.bound_x[0] > node.pos[0] or self.bound_x[1] < node.pos[0]:
            return False 
        if self.bound_y[0] > node.pos[1] or self.bound_y[1] < node.pos[1]:
            return False 
        return True
    
    def __str__(self):
        return(f"{self.constraints} \n {len(self.inputs)} \n {len(self.outputs)}")
    


    @property
    def bound(self):
        return self.constraints['location']
    @property
    def bound_x(self):
        return self.constraints['location'][0]
    @property
    def bound_y(self):
        return self.constraints['location'][1]

    """
    Constraint checking methods
    """
    def check_location_constraint(self, pos):
        """Check if the position of a node satisfies the location constraint

        @param pos the verfied againt the constraint
        """
        if self.bound[0][0] > pos[0] or self.bound[0][1] < pos[0]:
            return False 
        if self.bound[1][0] > pos[1] or self.bound[1][1] < pos[1]:
            return False 
        return True


    
    def __str__(self):
        return(f"Task {self.taskId} at: {self.bound} on node {self.node}")
