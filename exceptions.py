class InvalidAssignmentException(Exception):
    """Thrown wehn apply_assignment is passed an invalid assignment"""
    pass 

class UnassignedTaskException(Exception):
    """ Exception to be raise then an unassigned task is encountered"""     
    pass

class NoValidNodeException(Exception):
    """ Exception to be raised when a task has no valid nodes to be assigned to """
    pass

class NodeDiedException(Exception):
    """ Exception to be raised whenever a node dies during the network update.

    Additinally collects some relevant information about the network state that can be accessed via 
    the exception attributes
    """
    def __init__(self, node_indexes, average_energy, max_energy):
        self.node_indexes = node_indexes
        self.average_energy = average_energy
        self.max_energy = max_energy

class NoValidAssignmentException(Exception):
    pass 

class InvalidNetworkException(Exception):
    pass


class NetworkDeadException(Exception):
    pass
