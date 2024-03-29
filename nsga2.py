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
from network import evaluate, checkIfAlive, evaluate_surrogate, remove_dead_nodes
from collections import defaultdict, namedtuple
import argparse
from individual import ListWithAttributes
import itertools
import multiprocessing as mp
import os
import exceptions
import math
import numpy.linalg as la
import copy 
try:
   from collections.abc import Sequence
except ImportError:
   from collections import Sequence





def random_allocation(networkGraph = None):
    return random.randint(0, len(networkGraph.nodes())-1)

def random_assignment(networkGraph = None, taskGraph = None, node_status = []):
    nodes = list(networkGraph.nodes())
    prev = 0
    assignment = []
    for i, task in enumerate(taskGraph.nodes()):
        try:
            valid_nodes = task.get_constrained_nodes(networkGraph)
            #enabled_valid_nodes = []
            #for valid_node in valid_nodes:
            #   if node_status[nodes.index(valid_node)]:
            #      enabled_valid_nodes.append(valid_node)
        except exceptions.NoValidNodeException as e:
            print("Could not find a valid node for a task while creating random assignment")
            raise e
        if len(valid_nodes) == 0:
            print("Valid nodes empty while creating random assignment")
            raise exceptions.NoValidNodeException
        if prev != 0:
          choices = []
          for i in range(4):
            choices.append(np.random.choice(valid_nodes))
          choices=[nodes.index(x) for x in choices]
          diffs = [abs(x-prev) for x in choices]
          choice = min(diffs)
          choice_index = choices[diffs.index(choice)]
          assignment.append(choice_index)
          prev = choice_index
        else:
          node = np.random.choice(valid_nodes)
          node_index = nodes.index(node)
          assignment.append(node_index)
          prev = node_index
    return assignment

def checkIfAllocValid(alloc, node_status):
   for x in alloc:
      if (not node_status[x]):
         return False
   return True

def getNewValidAssignment(alloc, node_status, archive, **kwargs):
   nNodes = kwargs['nNodes']
   network_creator = topologies.network_topologies[kwargs['network_creator']]
   nTasks = kwargs['nTasks']
   task_creator = topologies.task_topologies[kwargs['task_creator']]
   energy_list = kwargs['energy_list_sim']
   networkGraph = network_creator(energy_list = energy_list, **kwargs)
   taskGraph = task_creator(networkGraph, **kwargs)   
   aliveGraph = network_creator(energy_list = energy_list, **kwargs)
   remove_dead_nodes(aliveGraph, energy_list, energy_only=True, **kwargs)
   if len(archive) > 0:
      #valid_archive = [x for x in archive if checkIfAllocValid(x, node_status)]
      #if len(valid_archive) > 0:
     return tools.selNSGA2(valid_archive, 1)[0]
   nodes = list(aliveGraph.nodes())
   tasks = list(taskGraph.nodes())
   new_alloc = []
   for i,x in enumerate(alloc):
      if not(node_status[x]):
         #alloc invalid, change:
         try:
            valid_nodes = tasks[i].get_constrained_nodes(aliveGraph)
            #enabled_valid_nodes = []
            #for valid_node in valid_nodes:
               #if node_status[nodes.index(valid_node)]:
                  #enabled_valid_nodes.append(valid_node)
         except exceptions.NoValidNodeException as e:
            print("Could not find a valid node for a task while adjusting faulty assignment")
            raise e
         if len(valid_nodes) == 0:
            print("no enabled valid node while adjusting faulty assignment")
            raise exceptions.NoValidNodeException
         cur_node = nodes[x]
         new_node = None
         dist = np.inf
         for node in valid_nodes:
            new_dist = la.norm(node.pos - cur_node.pos)
            if new_dist < dist:
               dist = new_dist
               new_node = node
         if node is None:
            print("No node closer than inf found while adjusting faulty assignment")
            raise exceptions.NoValidNodeException
         new_alloc.append(nodes.index(new_node))
      else:
         new_alloc.append(x)
   if kwargs['verbose']:
      print(f"new valid alloc: {new_alloc}")
   return new_alloc

def getNewAllocation(pop, archive, node_status, **settings):
   valid_allocs = []
   poparchive = pop + archive
   for alloc in poparchive:
      if checkIfAllocValid(alloc, node_status):
         valid_allocs.append(alloc)
   if len(valid_allocs) > 0:
      if settings['algorithm'] == 'nsga2' or settings['algorithm'] == 'rmota' or settings['algorithm'] == 'mmota':
         pfront = sortEpsilonNondominated(valid_allocs, len(pop))[0]
         best = tools.selBest(pfront,1)
      else:
         best = tools.selBest(pop,1)
      return best[0]
   else:
      energy_list = settings['energy_list_sim']
      network_creator = topologies.network_topologies[settings['network_creator']]
      task_creator = topologies.task_topologies[settings['task_creator']]
      networkGraph = network_creator(energy_list = energy_list,**settings)
      taskGraph = task_creator(networkGraph, **settings)   
      aliveGraph = network_creator(energy_list = energy_list,**settings)
      remove_dead_nodes(aliveGraph, energy_list, **settings)
      try:
         pop, toolbox= setup_run(aliveGraph, taskGraph, 0, popSize=1, **settings)
      except exceptions.NoValidNodeException:
        raise exceptions.NetworkDeadException
      except exceptions.NetworkDeadException as e:
        raise e
      except Exception as e:
         print(f"Error during ea setup: {e}")
         raise e
      return pop[0]
   
def repair_individual(ind, networkGraph, taskGraph, node_status):
   tasks = list(taskGraph.nodes())
   nodes = list(networkGraph.nodes())
   for task_number, node_number in enumerate(ind):
        #print(task_number)
        assigned_task = tasks[task_number]
        try:
            if node_number >= len(nodes) or node_number <0:
                #print(f"node number out of bounds: {node_number}")
                valid_nodes = tasks[task_number].get_constrained_nodes(networkGraph)
                #enabled_valid_nodes = []
                #for valid_node in valid_nodes:
                #  if node_status[nodes.index(valid_node)]:
                #    enabled_valid_nodes.append(valid_node)
                if len(valid_nodes) == 0:
                   print("No valid nodes in repair_individual")
                   raise exceptions.NoValidNodeException
                node_number = nodes.index(np.random.choice(valid_nodes))      
                ind[task_number] = node_number
            assigned_node = nodes[node_number]
            if not assigned_task.check_bounds(assigned_node):
                valid_nodes = tasks[task_number].get_constrained_nodes(networkGraph)
                #print(f"valid nodes for {task_number}:")
                #for x in valid_nodes:
                  #print(list(networkGraph.nodes()).index(np.random.choice(valid_nodes)))

                #enabled_valid_nodes = []
                #for valid_node in valid_nodes:
                #  if node_status[nodes.index(valid_node)]:
                #    enabled_valid_nodes.append(valid_node)
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
    if k == 0:
        return []
    angle = 30
    a = math.tan(math.radians(angle*2))/2
    map_fit_ind = defaultdict(list)
    max_fit = [0,0,0]
    for ind in individuals:
        for i,fit in enumerate(ind.fitness.values):
          if abs(fit) > max_fit[i]:
            max_fit[i] = abs(fit)
    for ind in individuals:
        new_fit = creator.FitnessMin((ind.fitness.values[0]/max_fit[0]*1+ind.fitness.values[1]/max_fit[1]*a + ind.fitness.values[2]/max_fit[2]*a, ind.fitness.values[1]/max_fit[1]*1+ind.fitness.values[0]/max_fit[0]*a+ ind.fitness.values[2]/max_fit[2]*a, ind.fitness.values[2]/max_fit[2]*1 + ind.fitness.values[0]/max_fit[0]*a + ind.fitness.values[1]/max_fit[1]*a,))
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


def selArchive(pop, archive, archivestats, max_size = 100):
   for ind in pop:
      similarity = 0
      for alloc in set(ind):
         if alloc in archivestats.keys():
            similarity += archivestats[alloc]
            archivestats[alloc] +=1
         else:
            archivestats.update({alloc : 1})
      ind.similarity = similarity
      archive.append(ind)
   for ind in archive:
      similarity = 0
      for alloc in set(ind):
         if alloc in archivestats.keys():
            similarity += archivestats[alloc]
            #archivestats[alloc] +=1
         else:
            #okay this should never happen..but just to be sure
            archivestats.update({alloc : 1})
      ind.similarity = similarity
   new_archive = sorted(archive, key = lambda x : x.similarity)
   #update the stats
   for ind in new_archive[max_size:]:
      for alloc in set(ind):
         archivestats[alloc] -=1
   return new_archive[:max_size], archivestats

def setup_ea(networkGraph = None, taskGraph = None, energy = None, eta = 20,**kwargs):
   if kwargs['algorithm'] == 'nsga2' or kwargs ['algorithm'] =='rmota' or kwargs['algorithm'] == 'mmota':
      creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0,))
   elif kwargs['algorithm'] == 'dtas':
      creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
   else:
      print("unrecognized algorithm")
      return -1

   creator.create("Individual", ListWithAttributes, fitness=creator.FitnessMin)
   
poolSize = 52
popSize = 100
def setup_run(networkGraph = None, taskGraph = None, energy = None, eta = 20, archive=[], archivestats = {}, popSize = popSize, multiprocessed = False, **kwargs):
   toolbox = base.Toolbox()
   toolbox.register("assignment", random_assignment, networkGraph = networkGraph, taskGraph = taskGraph, node_status = kwargs['network_status'])
   toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.assignment)
   #toolbox.register("allocation", random_allocation, networkGraph = networkGraph, taskGraph = taskGraph)
   #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.allocation, n=len(taskGraph.nodes))
   toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   if multiprocessed:
      pool = mp.Pool(poolSize)
      toolbox.register("map", pool.map)
   else:
      toolbox.register("map", map)
   toolbox.register("evaluate", evaluate)
   BOUND_LOW, BOUND_UP = 0.0, len(networkGraph.nodes())
   NDIM = 2
   if kwargs['crossover'] == 'nsga2':
      crossover = mycxSimulatedBinaryBounded
      toolbox.register("mate", crossover, low=BOUND_LOW, up=BOUND_UP, eta=eta)
   elif kwargs['crossover'] == 'twopoint':
      crossover = tools.cxTwoPoint
      toolbox.register("mate", crossover)
   if kwargs['algorithm'] == 'nsga2' or kwargs['algorithm'] == 'rmota' or kwargs['algorithm'] =='mmota':
      toolbox.register("mutate", mutRandomNode, networkGraph = networkGraph, taskGraph= taskGraph, indpb=1.0/NDIM)
      toolbox.register("select", tools.selNSGA2)
      if kwargs['algorithm'] == 'rmota':
         toolbox.register("selArchive", selArchive)
   elif kwargs['algorithm'] == 'dtas':
      toolbox.register("mate", tools.cxOnePoint)
      toolbox.register("mutate", mutRandomNode, networkGraph = networkGraph, taskGraph= taskGraph, indpb=0.01)
      toolbox.register("select", tools.selRoulette)
   else:
      print("unrecognized algorithm")
      return -1
   #row_alloc = creator.Individual([i for i in range(kwargs['nTasks'])])
   #zero_alloc = creator.Individual([0]*nTasks)
   #rev_alloc = creator.Individual([nTasks-(i+1) for i in range(kwargs['nTasks'])])
   
   new_pop = toolbox.population(popSize-len(archive))
   print("seeding with")
   print(len(archive))
   if len(archive) >0:
      for ind in archive:
         repair_individual(ind, networkGraph, taskGraph, kwargs['network_status'])
         #ind.fitness.valid = False
   pop= archive + new_pop
      
   return pop, toolbox

def evaluate_wrapper(args):
   allocation  = args[0]
   settings = args[1]
   index = args[2]
   posList = [[x[0], x[1]] for x in settings['prediction_data']]
   if len(posList) > 0:
     confidence = min([x[2] for x in settings['prediction_data']])
     if settings['algorithm'] == 'mmota' and index < max(confidence * 100, 20):
       settings_predicted = copy.deepcopy(settings)
       settings_predicted.update({'node_data' : posList})
       lifetime, latency, nMissed, energy_list = evaluate(allocation, **settings_predicted)
   lifetime, latency, nMissed, energy_list = evaluate(allocation, **settings)
   return lifetime, latency, nMissed, energy_list

def assign_fitnesses(pop,fitnesses,algorithm):
  for ind, fit in zip(pop, fitnesses):
    if algorithm == 'nsga2':
      ind.fitness.values = fit[:3]
    elif algorithm == 'rmota':
      ind.fitness.values = fit[:3]
    elif algorithm == 'mmota':
      ind.fitness.values = fit[:3]
    elif algorithm == 'dtas':
      ind.fitness.values = fit[0],
    else:
      raise exceptions.UnrecognizedAlgorithmException
    ind.lifetime = fit[0]
    ind.latency = fit[1]
    ind.nMissed = fit[2]
    ind.energy = fit[3]
  return pop

def select_parents(pop, toolbox, n_selected, algorithm):
    if algorithm == 'nsga2' or algorithm == 'rmota' or algorithm == 'mmota':
        offspring = tools.selTournamentDCD(pop, len(pop))
    elif algorithm =='dtas':
        offspring = toolbox.select(pop, n_selected)
    else:
        raise exceptions.UnrecognizedAlgorithmException
    return offspring

def generate_offspring(offspring, toolbox, CXPB, algorithm):
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if algorithm =='nsga2' or algorithm =='rmota' or algorithm == 'mmota':
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
        elif algorithm=='dtas':
            toolbox.mate(ind1,ind2)
        else:
            raise exceptions.UnrecognizedAlgorithmException
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)
        del ind1.fitness.values, ind2.fitness.values
    return offspring

def main(archive = [], archivestats={}, **kwargs):
    for key,val in kwargs.items():
       print(f"{key} : {val}")
    seed = kwargs['seed'] 
    random.seed(seed)
    np.random.seed(seed)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    NGEN = kwargs['NGEN']
    MU = 100 #100
    CXPB = 0.9
    eta = 20
    n_selected = MU if (kwargs['algorithm'] == 'nsga2' or kwargs['algorithm'] == 'rmota' or kwargs['algorithm'] =='mmota') else int((MU*0.8)/2)
    algorithm = kwargs['algorithm']
    prediction = kwargs['prediction_data']
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    nNodes = kwargs['nNodes']
    network_creator = topologies.network_topologies[kwargs['network_creator']]
    nTasks = kwargs['nTasks']
    task_creator = topologies.task_topologies[kwargs['task_creator']]
    energy_list = kwargs['energy_list_sim']
    networkGraph = network_creator(**kwargs)
    taskGraph = task_creator(networkGraph, **kwargs)   
    
    aliveGraph = network_creator(**kwargs)
    remove_dead_nodes(aliveGraph, energy_list, **kwargs)
    try:
       pop, toolbox= setup_run(aliveGraph, taskGraph, eta, archive= archive, archivestats=archivestats, **kwargs)
    except exceptions.NoValidNodeException:
      raise exceptions.NetworkDeadException
    except exceptions.NetworkDeadException as e:
      raise e
    except Exception as e:
       print(f"Error during ea setup: {e}")
       raise e
    pop.sort(key=lambda x: x.fitness.values)
    
    #Evaluate the individuals with an invalid fitness
    #invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    mapped = zip(pop, itertools.repeat(kwargs),list(range(len(pop))))
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
    pop = assign_fitnesses(pop, fitnesses, algorithm) 
    #crowding distance assignment
    if algorithm == 'nsga2' or algorithm == 'mmota':
       pop = toolbox.select(pop, len(pop))
    elif algorithm == 'rmota':
       pop = toolbox.select(pop, len(pop))
       archive, archivestats = toolbox.selArchive(pop, archive, archivestats, popSize)

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)
    ################################
    # Begin the generational process
    ###############################
    for gen in range(1, NGEN):
        parents = select_parents(pop, toolbox, n_selected, algorithm)
        offspring = [toolbox.clone(ind) for ind in parents]
        offspring = generate_offspring(offspring, toolbox, CXPB, algorithm)
        # Evaluate the individuals with an invalid fitness
        for ind in offspring:
           repair_individual(ind, aliveGraph, taskGraph, kwargs['network_status'])
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        mapped = zip(pop, itertools.repeat(kwargs),list(range(len(pop))))
        try:
          fitnesses = toolbox.map(evaluate_wrapper, mapped)
        except exceptions.NoValidNodeException:
          raise exceptions.NetworkDeadException
        except exceptions.NetworkDeadException as e:
          raise e
        except Exception as e:
           print(f"error during ea loop: {e}")
           raise e
        assign_fitnesses(invalid_ind, fitnesses, algorithm) 
        # Select the next generation population
        pop = toolbox.select(pop + offspring, len(pop))
        if algorithm == 'rmota':
           archive, archivestats = toolbox.selArchive(pop, archive, archivestats, popSize)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
    
    if kwargs['algorithm'] == 'nsga2' or kwargs['algorithm'] =='rmota' or kwargs['algorithm'] == 'mmota':
      #pfront = sortEpsilonNondominated(pop, len(pop))[0]
      if algorithm =='mmota' and len(kwargs['prediction_data']) > 0:
        confidence = min([x[2] for x in kwargs['prediction_data']])
        predN = max(confidence*100,20)
        pfront1 = tools.sortNondominated(pop[:predN], predN, True)[0]
        best1 = tools.selBest(pfront1,1)
        pfront2 = tools.sortNondominated(pop[predN:], len(pop-predN), True)[0]
        best2 = tools.selBest(pfront2,1)
        posList = [[x[0], x[1]] for x in kwargs['next_state']]
        settings_predicted = copy.deepcopy(settings)
        settings_predicted.update({'node_data' : posList})
        settings_predicted.update({'prediction_data' : []})
        mapped = zip(pop, itertools.repeat(settings_predicted),[0,1])
        try:
          fitnesses = toolbox.map(evaluate_wrapper, mapped)
        except exceptions.NoValidNodeException:
          raise exceptions.NetworkDeadException
        except exceptions.NetworkDeadException as e:
          raise e
        except Exception as e:
           print(f"Error during ea finish: {e}")
           raise e
        assign_fitnesses([best1, best2], fitnesses, 'mmota')
      else:
        pfront = tools.sortNondominated(pop, len(pop), True)[0]
        best = tools.selBest(pfront,1)
    else:
      best = tools.selBest(pop,1)
    if kwargs['algorithm'] == 'nsga2' or kwargs['algorithm'] == 'dtas' or kwargs['algorithm'] == 'mmota':
      archive = pop#tools.sortNondominated(pop, len(pop), True)[0]
      
    return pop, logbook, best, archive, archivestats


def save(index, db, bests, fronts, settings):
    import sqlalchemy as sql
    import pandas as pd 
    import json
    import pickle
    old_results = pd.read_sql("results_mobility", con=db)
    #min_index = old_results.index.max() + 1 if len(old_results) > 0 else 0
    results = {'index' : index,
               'bests' : pickle.dumps(bests),
               'fronts' : pickle.dumps(fronts),
               'settings' : json.dumps(settings),
               'algorithm' : settings['algorithm'],
               'nnodes' : settings['nNodes'],
               'ntasks' : settings['nTasks'],
               'static' : settings['static'],
               'predictor' : settings['predictor']
               }
    df = pd.DataFrame(results, index=[index])
    df.set_index('index', inplace=True)
    print(df)
    df.to_sql('results_mobility', db, if_exists='append')
    with open(f"{settings['datapath'][:-5]}_bests{index}.pck","wb+") as f:
      pickle.dump(bests, f)
    with open(f"{settings['datapath'][:-5]}_fronts{index}.pck","wb+") as f:
      pickle.dump(fronts, f)
    print(bests)  
    for x in bests:
      print(x.lifetime)
      print(x.latency)
      print(x.nMissed)

def run_algorithm(index, db, **settings):  
    import sqlalchemy as sql
    import pandas as pd 
    import json
    import pickle
    bests = []
    objectives = []
    recursive = True
    fronts = []
    setup_ea(**settings)
    algorithm = settings['algorithm']
    print(settings)
    node_data = []
    prediction_data = []
    with open(settings['datapath']) as f:
      node_data = json.load(f)
    
    with open(settings['predpath']) as f:
      prediction_data = json.load(f)
    #inital run:
    print(node_data[0])
    settings.update({'network_status' : node_data[0]})
    settings.update({'posList' : node_data[0]})
    settings.update({'prediction_data' : []})
    enl = settings['energy_list']

    settings.update({'energy_list_sim' : enl})
    pop, stats, best, archive, archivestats = main(archive = [], archivestats = {}, **settings)
    best = best[0]
    bests.append(best)
    front = tools.sortNondominated(pop, len(pop), True)[0]
    fronts.append(front)
    #new_lifetime, new_latency, new_received, new_energy_list, new_node_status, new_missed = evaluate(best, **settings)
    nl, l, nmissed, new_energy_list = evaluate(best, **settings)
    nsteps = min([len(node_data), len(prediction_data)])-2
    nsteps = min([nsteps,20])
    try:
      for i in range(1,nsteps):
        print(f"planning reallocations: {nsteps-i} to go" )
        settings.update({'energy_list_sim' : new_energy_list})
        settings.update({'network_status' : node_data[i]})
        settings.update({'posList' : node_data[i]})
        settings.update({'prediction' : prediction_data[i]}) 
        print(node_data[i])
        print(prediction_data[i])
        NGEN = settings['NGEN_realloc'] if 'NGEN_realloc' in settings.keys() else 10
        settings.update({'NGEN' : NGEN})
        
        pop, stats, best, archive, archivestats = main(archive = archive, archivestats=archivestats, **settings)
        best = best[0]
        bests.append(best)
        front = tools.sortNondominated(pop, len(pop), True)[0]
        fronts.append(front)
    except exceptions.NetworkDeadException:
        save(index, db, bests, fronts, settings)
    save(index, db, bests, fronts, settings)














if __name__ == "__main__":
  import sqlalchemy as sql
  nNodes = 81
  mobileNodes = 0
  dims = 7
  energy = 100
  network_creator = topologies.ManHattan
  task_creator = None
  nTasks = 19
  if network_creator == topologies.Grid or network_creator==topologies.ManHattan:
      nNodes = dims**2
  if network_creator == topologies.Line:
      dims = nNodes
  nNodes = nNodes + mobileNodes
  energy_list = [energy]*nNodes
  task_creator = 'EncodeDecode'
  network_creator = 'Manhattan'
  algorithm = 'mmota'
  for i in range(11):
    settings = {'nNodes' : nNodes,
               'mobileNodeCount' : mobileNodes,
               'network_creator' : network_creator,
               'dimx' : dims,
               'dimy' : dims,
               'deltax' :100,
               'deltay': 100,
               'nTasks' : nTasks,
               'task_creator' : task_creator,
               'energy_list' : energy_list ,
               'energy_list_sim' : energy_list ,
               'posList' : [],
               'init_energy' : energy,
               'verbose' : False,
               'capture_packets' : False,
               'pcap_filename' : f"pcap_minimal_network_{nTasks}task",
               'enable_errors' : False,
               'seed' : 3141 + i*21,
               'error_shape' : 1.0,
               'error_scale' : 1.0,
               'routing' : True,
               'static' : True,
               'run_number' : i,
               'predictor' : 'perfect',
               'datapath' : f"datasets/mobile/81/perfect/positions_{i}.json",
               'predpath' : f"datasets/mobile/81/perfect/predictions_{i}.json"
               }
  settings.update({'nTasks' : nTasks})
  settings.update({'crossover' : 'nsga2'})
  settings.update({'experiment' : 'mmota'})
  settings.update({'task_creator' : task_creator})
  settings.update({'algorithm' : algorithm})
  settings.update({'NGEN_realloc' : 2})
  settings.update({'NGEN' : 2})
  db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
  run_algorithm(500, db, **settings)
