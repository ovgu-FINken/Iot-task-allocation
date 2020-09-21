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
import numpy as np
from math import sqrt
import networkx as nx
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
import topologies
from network import evaluate, checkIfAlive, remove_dead_nodes
from collections import defaultdict, namedtuple

from individual import ListWithAttributes
import itertools
import multiprocessing as mp
import os
import exceptions
import math

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
        except exceptions.NoValidNodeException as e:
            print("Could not find a valid node for a task while creating random assignment")
            raise e
        if len(valid_nodes) == 0:
            print("Valid nodes empty while creating random assignment")
            raise exceptions.NoValidNodeException
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
         except exceptions.NoValidNodeException as e:
            print("Could not find a valid node for a task while mutating assignment")
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
            individual[i] = int(round(x))
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

def sortEpsilonNondominated(individuals, k, first_front_only=False):
    """Sort the first *k* *individuals* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    individuals.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param first_front_only: If :obj:`True` sort only the first front and
                             exit.
    :returns: A list of Pareto fronts (lists), the first list includes
              nondominated individuals.
    .. [Deb2002] Deb, Pratab, Agarwal, and Mearivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []
    angle = 30
    a = math.tan(math.radians(angle*2))/2
    map_fit_ind = defaultdict(list)
    for ind in individuals:
        new_fit = creator.FitnessMin((ind.fitness.values[0]*1+ind.fitness.values[1]*a/1000, ind.fitness.values[1]*1/1000+ind.fitness.values[0]*a))
        map_fit_ind[new_fit].append(ind)
    fits = map_fit_ind.keys()

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in list(fits)[i+1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

def setup_ea(networkGraph = None, taskGraph = None, energy = None, eta = 20, **kwargs):
   if kwargs['algorithm'] == 'nsga2':
      creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
   elif kwargs['algorithm'] == 'dtas':
      creator.create("FitnessMin", base.Fitness, weights=(1.0,))
   else:
      print("unrecognized algorithm")
      return -1
   creator.create("Individual", ListWithAttributes, fitness=creator.FitnessMin)

   toolbox = base.Toolbox()
   toolbox.register("assignment", random_assignment, networkGraph = networkGraph, taskGraph = taskGraph)
   toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.assignment)
   #toolbox.register("allocation", random_allocation, networkGraph = networkGraph, taskGraph = taskGraph)
   #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.allocation, n=len(taskGraph.nodes))
   toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   pool = mp.Pool(65)
   toolbox.register("map", pool.map)
   toolbox.register("evaluate", evaluate)
   BOUND_LOW, BOUND_UP = 0.0, len(networkGraph.nodes())
   NDIM = 2
   if kwargs['algorithm'] == 'nsga2':
      toolbox.register("mate", mycxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=eta)
      toolbox.register("mutate", mutRandomNode, networkGraph = networkGraph, taskGraph= taskGraph, indpb=1.0/NDIM)
      toolbox.register("select", tools.selNSGA2)
   elif kwargs['algorithm'] == 'dtas':
      toolbox.register("mate", tools.cxOnePoint)
      toolbox.register("mutate", mutRandomNode, networkGraph = networkGraph, taskGraph= taskGraph, indpb=0.01)
      toolbox.register("select", tools.selRoulette)
   else:
      print("unrecognized algorithm")
      return -1

   pop = toolbox.population(100)
   return pop, toolbox

def evaluate_wrapper(args):
   allocation  = args[0]
   settings = args[1]
   time, latency, received, energy_list = evaluate(allocation, **settings)
   return time, latency, received, energy_list


def main(seed=None, **kwargs):
    random.seed(seed)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    NGEN = 200 #250
    MU = 100 #100
    CXPB = 0.9
    eta = 20
    n_selected = MU if kwargs['algorithm'] == 'nsga2' else int((MU*0.8)/2)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    nNodes = kwargs['nNodes']
    network_creator = kwargs['network_creator']
    nTasks = kwargs['nTasks']
    task_creator = kwargs['task_creator']
    energy_list = kwargs['energy_list']
    networkGraph = network_creator(**kwargs)
    taskGraph = task_creator(networkGraph, **kwargs)   
    
    aliveGraph = network_creator(**kwargs)
    remove_dead_nodes(aliveGraph, energy_list, **kwargs)
    try:
      pop, toolbox= setup_ea(aliveGraph, taskGraph, eta, **kwargs)
    except exceptions.NoValidNodeException:
      raise exceptions.NetworkDeadException
    except exceptions.NetworkDeadException as e:
      raise e
    except Exception as e:
       print(f"Error during ea setup: {e}")
       raise e
    pop.sort(key=lambda x: x.fitness.values)
    
    #Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    mapped = zip(invalid_ind, itertools.repeat(kwargs))
    #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, itertools.repeat([networkGraph,taskGraph, [energy]*25]))
    continue_run = checkIfAlive(**kwargs)
    if not continue_run:
       raise exceptions.NetworkDeadException
    try:
      fitnesses = toolbox.map(evaluate_wrapper, mapped)
    except exceptions.NoValidNodeException:
      raise exceptions.NetworkDeadException
    except exceptions.NetworkDeadException as e:
      raise e
    except Exception as e:
       print(f"Error during ea gen: {e}")
       raise e
    for ind, fit in zip(invalid_ind, fitnesses):
       if kwargs['algorithm'] == 'nsga2':
         ind.fitness.values = fit[:2]
       elif kwargs['algorithm'] == 'dtas':
         ind.latency = fit[1]
         if ind.latency > len(taskGraph.nodes())*1000:
            ind.fitness.values = -fit[0]-ind.latency/1000,
         else:
            ind.fitness.values = -fit[0],
       else:
          print("unrecognized algorithm")
          return -1
       ind.received = fit[2]
       ind.energy = fit[3]
   # # This is just to assign the crowding distance to the individuals
   # # no actual selection is done
    if kwargs['algorithm'] == 'nsga2':
       pop = toolbox.select(pop, len(pop))
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        if kwargs['algorithm'] == 'nsga2':
            offspring = tools.selTournamentDCD(pop, len(pop))
        else:
            offspring = toolbox.select(pop, n_selected)
            
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if kwargs['verbose']:
                print(f"parents: {ind1}  ,  {ind2}")
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            if kwargs['verbose']:
               print(f"children: {ind1}  ,  {ind2}")
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            if kwargs['verbose']:
               print(f"mutated: {ind1}  ,  {ind2}")
            del ind1.fitness.values, ind2.fitness.values
        # Evaluate the individuals with an invalid fitness
        for ind in offspring:
           repair_individual(ind, aliveGraph, taskGraph)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        mapped = zip(invalid_ind, itertools.repeat(kwargs))
        #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, itertools.repeat([networkGraph,taskGraph,[energy]*25]))
        try:
          fitnesses = toolbox.map(evaluate_wrapper, mapped)
        except exceptions.NoValidNodeException:
          raise exceptions.NetworkDeadException
        except exceptions.NetworkDeadException as e:
          raise e
        except Exception as e:
           print(f"error during ea loop: {e}")
           raise e
        for ind, fit in zip(invalid_ind, fitnesses):
            if kwargs['algorithm'] == 'nsga2':
               ind.fitness.values = fit[:2]
            elif kwargs['algorithm'] == 'dtas':
               ind.latency = fit[1]
               if ind.latency > len(taskGraph.nodes())*1000:
                  ind.fitness.values = -fit[0]-ind.latency/1000,
               else:
                  ind.fitness.values = -fit[0],
            else:
               print("unrecognized algorithm")
               return -1
            ind.received = fit[2]
            ind.energy = fit[3]
        # Select the next generation population
        pop = toolbox.select(pop + offspring, len(pop))
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    #print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))
    if kwargs['algorithm'] == 'nsga2':
      pfront = sortEpsilonNondominated(pop, len(pop))[0]
      best = tools.selBest(pfront,1)
    else:
      best = tools.selBest(pop,1)
    return pop, logbook, best
        
if __name__ == "__main__":
   import pickle
   import time

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
   
   nNodes = 20
   nTasks = 11
   dims = 9
   energy = 3
   algorithm = 'nsga2'
   network_creator = topologies.Grid
   task_creator = topologies.TwoTaskWithProcessing
   if network_creator == topologies.Grid:
      nNodes = dims**2
   if task_creator == topologies.EncodeDecode:
      nTasks= 19
   if task_creator == topologies.TwoTaskWithProcessing:
      nTasks = 20
   energy_list = [energy]*nNodes

   settings = {'nNodes' : nNodes,
             'network_creator' : network_creator,
             'dimx' : dims,
             'dimy' : dims,
             'nTasks' : nTasks,
             'task_creator' : task_creator,
             'energy_list' : energy_list ,
             'init_energy' : energy,
             'algorithm' : algorithm,
             'verbose' : False
             }

   def min2digits(a):
    s = str(a)
    if len(s) < 2:
       s = "0" + str(a)
    return s

   if not os.path.exists(f"results/{algorithm}/{network_creator.__name__}/{task_creator.__name__}/"):
       os.makedirs(f"results/{algorithm}/{network_creator.__name__}/{task_creator.__name__}/")
   seed = 2002
   offset = 42
   for i in range(11):
    print(f"Beginning iteration {i}")
    settings = {'nNodes' : nNodes,
             'network_creator' : network_creator,
             'dimx' : dims,
             'dimy' : dims,
             'nTasks' : nTasks,
             'task_creator' : task_creator,
             'energy_list' : energy_list ,
             'init_energy' : energy,
             'verbose' : False,
             'algorithm' : algorithm
             }
    print(f"Settings:")
    for key,value in settings.items():
       print(f"{key} : {value}")
    if os.path.isfile(f"results/{algorithm}/{network_creator.__name__}/{task_creator.__name__}/stats_nodes{min2digits(nNodes)}_tasks{min2digits(nTasks)}_{min2digits(i)}_00.pck"):
       continue
    start = time.time()
    bests = []
    objectives = []
    recursive = True
    j = 0
    if recursive:
       while True:
          try:
             print("beginning new run")
             runSeed = seed + i*offset
             settings.update({"prefix" : runSeed})
             pop, stats, best = main(seed = runSeed, **settings)
             best = best[0]
             new_energy_list = best.energy
             print(new_energy_list)
             settings.update({'energy_list' : new_energy_list})
             bests.append(list(best))
             if algorithm  == 'nsga2':
               objectives.append(best.fitness.values)
             else:
               objectives.append((best.fitness.values[0], best.latency))

             print(bests)
             print(objectives)
             print()
             with open(f"results/{algorithm}/{network_creator.__name__}/{task_creator.__name__}/stats_nodes{min2digits(nNodes)}_tasks{min2digits(nTasks)}_{min2digits(i)}_{min2digits(j)}.pck", "wb") as f:
               pickle.dump(stats, f)
             with open(f"results/{algorithm}/{network_creator.__name__}/{task_creator.__name__}/pop_nodes{min2digits(nNodes)}_tasks{min2digits(nTasks)}_{min2digits(i)}_{min2digits(j)}.pck", "wb") as f:
               pickle.dump(pop, f)
             with open(f"results/{algorithm}/{network_creator.__name__}/{task_creator.__name__}/objectives_nodes{min2digits(nNodes)}_tasks{min2digits(nTasks)}_{min2digits(i)}_{min2digits(j)}.pck", "wb") as f:
               pickle.dump(objectives, f)
             with open(f"results/{algorithm}/{network_creator.__name__}/{task_creator.__name__}/bests_nodes{min2digits(nNodes)}_tasks{min2digits(nTasks)}_{min2digits(i)}_{min2digits(j)}.pck", "wb") as f:
              pickle.dump(bests, f)
             j += 1
          except exceptions.NetworkDeadException:
             print(f"Time Elapsed for iteration {i}: {time.time() - start}")
             #print(stats)
             break
    else:
      runSeed = seed + i*offset
      pop, stats, best = main(seed = runSeed, **settings)
      with open(f"results/{algorithm}/{network_creator.__name__}/{task_creator.__name__}/stats_nodes{min2digits(nNodes)}_tasks{min2digits(nTasks)}_{min2digits(i)}.pck", "wb") as f:
         pickle.dump(stats, f)
      with open(f"results/{algorithm}/{network_creator.__name__}/{task_creator.__name__}/pop_nodes{min2digits(nNodes)}_tasks{min2digits(nTasks)}_{min2digits(i)}.pck", "wb") as f:
         pickle.dump(pop, f)
      print(f"{time.ctime()}: Iteration {i} finished, time elapsed: {time.time() - start}")

          

   #print("Convergence: ", convergence(pop, optimal_front))
   #print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

   #import matplotlib.pyplot as plt

   #front = np.array([ind.fitness.values for ind in pop])
   #optimal_front = np.array(optimal_front)
   #plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
   #plt.scatter(front[:,0], front[:,1], c="b")
   #plt.axis("tight")
   #plt.show()
