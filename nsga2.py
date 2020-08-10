#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
import sys
import array
import random
import json
import numpy as np
from math import sqrt
import networkx as nx
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from network import GraphNode, Grid, Line, TwoTask, TwoTaskWithProcessing, evaluate
import itertools
from exceptions import NoValidNodeException
import multiprocessing as mp

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

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence






def random_allocation(networkGraph = None):
    return random.randint(0, len(networkGraph.nodes())-1)

def random_assignment(networkGraph = None, taskGraph = None):
    nodes = list(networkGraph.nodes())
    assignment = []
    for i, task in enumerate(taskGraph.nodes()):
        try:
            valid_nodes = task.get_constrained_nodes(networkGraph)
        except NoValidNodeException as e:
            print("Could not find a valid node for a task while creating random assignment")
            raise e
        node = np.random.choice(valid_nodes)
        node_index = nodes.index(node)
        assignment.append(node_index)
    return assignment


def repair_individual(ind, networkGraph, taskGraph):
    for task_number, node_number in enumerate(ind):
        assigned_task = list(taskGraph.nodes())[task_number]
        try:
            if node_number >= len(networkGraph.nodes()) or node_number <0:
                valid_nodes = list(taskGraph.nodes())[task_number].get_constrained_nodes(networkGraph)
                node_number = networkGraph.nodes.index(np.random.choice(valid_nodes))      
                ind[task_number] = node_number
            assigned_node = list(networkGraph.nodes())[node_number]
            if not assigned_task.check_bounds(assigned_node):
                valid_nodes = list(taskGraph.nodes())[task_number].get_constrained_nodes(networkGraph)
                node_number = list(networkGraph.nodes()).index(np.random.choice(valid_nodes))
                ind[task_number] =node_number
        except IndexError as e:
            print(f"Encountered invalid indexing: task_number : {task_number}, node_number : {node_number}")
            raise
    return ind


def mutRandomNode(individual, networkGraph = None, taskGraph= None, indpb= 0):
   nodes = list(networkGraph.nodes())
   for i, task in enumerate(taskGraph.nodes()):
      if random.random() <=indpb:
         try:
            valid_nodes = task.get_constrained_nodes(networkGraph)
         except NoValidNodeException as e:
            print("Could not find a valid node for a task while creating random assignment")
            raise e
         node = np.random.choice(valid_nodes)
         node_index = nodes.index(node)
         individual[i] = node_index
   return individual, 


def mymutPolynomialBounded(individual, eta, low, up, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = itertools.repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = itertools.repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = int(x)
    return individual,

def mycxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    """Executes a simulated binary crossover that modify in-place the input
    individuals. The simulated binary crossover expects :term:`sequence`
    individuals of floating point numbers.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :param low: A value or a :term:`python:sequence` of values that is the lower
                bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that is the upper
               bound of the search space.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    .. note::
       This implementation is similar to the one implemented in the
       original NSGA-II C code presented by Deb.
    """
    size = min(len(ind1), len(ind2))
    if not isinstance(low, Sequence):
        low = itertools.repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of the shorter individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = itertools.repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of the shorter individual: %d < %d" % (len(up), size))
    #print(f"Before Crossover: {ind1} , {ind2}")
    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= 0.5:
            # This epsilon should probably be changed for 0 since
            # floating point arithmetic in Python is safer
            if abs(ind1[i] - ind2[i]) > 1e-14:
                x1 = min(ind1[i], ind2[i])
                x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if random.random() <= 0.5:
                    ind1[i] = int(c2)
                    ind2[i] = int(c1)
                else:
                    ind1[i] = int(c1)
                    ind2[i] = int(c2)

    #print(f"After Crossover: {ind1} , {ind2}")
    return ind1, ind2


def setup_ea(networkGraph = None, taskGraph = None):
   creator.create("FitnessMixed", base.Fitness, weights=(-1.0, -1.0))
   creator.create("Individual", list, fitness=creator.FitnessMixed)

   toolbox = base.Toolbox()
   toolbox.register("assignment", random_assignment, networkGraph = networkGraph, taskGraph = taskGraph)
   toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.assignment)
   #toolbox.register("allocation", random_allocation, networkGraph = networkGraph, taskGraph = taskGraph)
   #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.allocation, n=len(taskGraph.nodes))
   toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   pool = mp.Pool()
   toolbox.register("map", pool.starmap)
   toolbox.register("evaluate", evaluate)
   BOUND_LOW, BOUND_UP = 0.0, len(networkGraph.nodes())
   NDIM = 2
   toolbox.register("mate", mycxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
   toolbox.register("mutate", mutRandomNode, networkGraph = networkGraph, taskGraph= taskGraph, indpb=0.25)
   toolbox.register("select", tools.selNSGA2)
   pop = toolbox.population(60)
   return pop, toolbox



def main(seed=None, energy = 250, nNodes = 25):
    random.seed(seed)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    NGEN = 250
    MU = 100
    CXPB = 0.9
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    networkGraph = Line(nNodes)
    taskGraph = TwoTaskWithProcessing(networkGraph)
    pop, toolbox = setup_ea(networkGraph, taskGraph)
    pop.sort(key=lambda x: x.fitness.values)
    

    #Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    mapped = zip(invalid_ind, itertools.repeat([networkGraph,taskGraph, [energy]*nNodes]))
    #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, itertools.repeat([networkGraph,taskGraph, [energy]*25]))
    fitnesses = toolbox.map(toolbox.evaluate, mapped)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
   # # This is just to assign the crowding distance to the individuals
   # # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), pop=list(pop), **record)
    print(logbook.stream)
    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        # Evaluate the individuals with an invalid fitness
        for ind in offspring:
           repair_individual(ind, networkGraph, taskGraph)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        mapped = zip(invalid_ind, itertools.repeat([networkGraph,taskGraph, [energy]*25]))
        #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, itertools.repeat([networkGraph,taskGraph,[energy]*25]))
        fitnesses = toolbox.map(toolbox.evaluate, mapped)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), pop=list(pop), **record)
        print(logbook.stream)

    #print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook
        
if __name__ == "__main__":
    import pickle
    import time
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
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    start = time.time()
    pop, stats = main(energy = 10, nNodes = 20)
    print(f"Time Elapsed: {time.time() - start}")
    print(stats)
    with open("stats.pck", "wb") as f:
      pickle.dump(stats, f)
    with open("pop.pck", "wb") as f:
      pickle.dump(pop, f)

    #print("Convergence: ", convergence(pop, optimal_front))
    #print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    import matplotlib.pyplot as plt

    front = np.array([ind.fitness.values for ind in pop])
    #optimal_front = np.array(optimal_front)
    #plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()
