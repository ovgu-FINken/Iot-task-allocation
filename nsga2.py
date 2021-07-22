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

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence





def random_allocation(networkGraph = None):
    return random.randint(0, len(networkGraph.nodes())-1)

def random_assignment(networkGraph = None, taskGraph = None, node_status = []):
    nodes = list(networkGraph.nodes())
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
        node = np.random.choice(valid_nodes)
        node_index = nodes.index(node)
        assignment.append(node_index)
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
      valid_archive = [x for x in archive if checkIfAllocValid(x, node_status)]
      if len(valid_archive) > 0:
         return tools.selNSGA2(valid_archive, 1)[0]
   nodes = list(aliveGraph.nodes())
   tasks = list(taskGraph.nodes())
   new_alloc = []
   for i,x in enumerate(alloc):
      if not(node_status[x]):
         #alloc invalid, change:
         try:
            valid_nodes = tasks[i].get_constrained_nodes(aliveGraph)
            enabled_valid_nodes = []
            for valid_node in valid_nodes:
               if node_status[nodes.index(valid_node)]:
                  enabled_valid_nodes.append(valid_node)
         except exceptions.NoValidNodeException as e:
            print("Could not find a valid node for a task while adjusting faulty assignment")
            raise e
         if len(enabled_valid_nodes) == 0:
            print("no enabled valid node while adjusting faulty assignment")
            raise exceptions.NoValidNodeException
         cur_node = nodes[x]
         new_node = None
         dist = np.inf
         for node in enabled_valid_nodes:
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
      if settings['algorithm'] == 'nsga2' or settings['algorithm'] == 'rmota':
         pfront = sortEpsilonNondominated(pop, len(pop))[0]
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
        assigned_task = tasks[task_number]
        try:
            if node_number >= len(nodes) or node_number <0:
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
    for ind in individuals:
        new_fit = creator.FitnessMin((ind.fitness.values[0]*1/5+ind.fitness.values[1]*a, ind.fitness.values[1]*1+ind.fitness.values[0]*a/5))
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
   for ind in archive:
      similarity = 0
      for alloc in set(ind):
         if alloc in archivestats.keys():
            similarity += archivestats[alloc]
            archivestats[alloc] +=1
         else:
            archivestats.update({alloc : 1})
      ind.similarity = similarity
   new_archive = sorted(pop+archive, key = lambda x : x.similarity)
   #update the stats
   for ind in new_archive[max_size:]:
      for alloc in set(ind):
         archivestats[alloc] -=1
   return new_archive[:max_size], archivestats

def setup_ea(networkGraph = None, taskGraph = None, energy = None, eta = 20,**kwargs):
   if kwargs['algorithm'] == 'nsga2' or kwargs ['algorithm'] =='rmota':
      creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
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
   if kwargs['algorithm'] == 'nsga2' or kwargs['algorithm'] == 'rmota':
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
   if len(archive) >0:
      for ind in archive:
         repair_individual(ind, networkGraph, taskGraph, kwargs['network_status'])
         #ind.fitness.valid = False
   pop= archive + new_pop
      
   return pop, toolbox

def evaluate_wrapper(args):
   allocation  = args[0]
   settings = args[1]
   lifetime, latency, received, energy_list, node_status, missed = evaluate_surrogate(allocation, **settings)
   return lifetime, latency, received, energy_list, node_status

def assign_fitnesses(pop,fitnesses,algorithm):
   for ind, fit in zip(pop, fitnesses):
           if algorithm == 'nsga2':
               ind.fitness.values = fit[:2]
           elif algorithm == 'rmota':
               ind.fitness.values = fit[:2]
           elif algorithm == 'dtas':
               ind.fitness.values = fit[0],
           else:
               raise exceptions.UnrecognizedAlgorithmException
       ind.latency = fit[1]
       ind.received = fit[2]
       ind.energy = fit[3]
       ind.node_status=fit[4]
    return pop
def select_parents(pop, toolbox, n_selected, algorithm):
    if algorithm == 'nsga2' or algorithm == 'rmota':
        offspring = tools.selTournamentDCD(pop, len(pop))
    elif algorithm =='dtas':
        offspring = toolbox.select(pop, n_selected)
    else:
        raise exceptions.UnrecognizedAlgorithmException
    return offspring

def generate_offspring(offspring, toolbox, CXPB):
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if algorithm =='nsga2' or algorithm =='dtas':
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
        elif algorithm=='dtas':
            toolbox.mate(ind1,ind2)
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
    n_selected = MU if (kwargs['algorithm'] == 'nsga2' or kwargs['algorithm'] == 'rmota') else int((MU*0.8)/2)
    algorithm = kwargs['algorithm']

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    nNodes = kwargs['nNodes']
    network_creator = topologies.network_topologies[kwargs['network_creator']]
    nTasks = kwargs['nTasks']
    task_creator = topologies.task_topologies[kwargs['task_creator']]
    energy_list = kwargs['energy_list_sim']
    networkGraph = network_creator(energy_list=energy_list, **kwargs)
    taskGraph = task_creator(networkGraph, **kwargs)   
    
    aliveGraph = network_creator(energy_list=energy_list,**kwargs)
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
    mapped = zip(pop, itertools.repeat(kwargs))
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
    if algorithm == 'nsga2'
       pop = toolbox.select(pop, len(pop))
    elif 'algorithm' == 'rmota':
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
        offspring = [toolbox.clone(ind) for ind in offspring]
        offspring = generate_offspring(offspring, toolbox, CXPB)
        # Evaluate the individuals with an invalid fitness
        for ind in offspring:
           repair_individual(ind, aliveGraph, taskGraph, kwargs['network_status'])
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        mapped = zip(invalid_ind, itertools.repeat(kwargs))
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
    
    if kwargs['algorithm'] == 'nsga2' or kwargs['algorithm'] =='rmota':
      pfront = sortEpsilonNondominated(pop, len(pop))[0]
      best = tools.selBest(pfront,1)
    else:
      best = tools.selBest(pop,1)
    if kwargs['algorithm'] == 'nsga2' or kwargs['algorithm'] == 'dtas':
      archive = []
    return pop, logbook, best, archive, archivestats

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
      
    actual_times = []
    actual_latencies = []
    missed_packages = []
    algorithm = settings['algorithm']
    #inital run:
    pop, stats, best, archive, archivestats = main(archive = [], archivestats = {}, **settings)
    best = best[0]
    settings.update({'enable_errors' : True})
    new_lifetime, new_latency, new_received, new_energy_list, new_node_status, new_missed = evaluate(best, **settings)
    actual_times.append(new_lifetime)
    actual_latencies.append(new_latency)
    settings.update({'energy_list_sim' : new_energy_list})
    settings.update({'network_status' : new_node_status})
    NGEN = settings['NGEN_realloc'] if 'NGEN_realloc' in settings.keys() else 10
    if settings['algorithm']=='dtas':
       NGEN=int(NGEN*2.5)
    settings.update({'NGEN' : NGEN})
    bests.append(list(best))
    front = tools.sortNondominated(pop, len(pop), True)[0]
    fronts.append(front)
    if algorithm  == 'nsga2' or algorithm =='rmota':
      objectives.append(best.fitness.values)
    else:
      objectives.append((best.fitness.values[0], best.latency))
    run_number = 2
    while True:
      try:
        settings.update({'run_number' : run_number})
        try:
          new_alloc = getNewValidAssignment(best, new_node_status, archive, **settings)
        except exceptions.NoValidNodeException:
          raise exceptions.NetworkDeadException

        if len(archive) == 0:
          #print("arhcive len 0!")
          new_alloc=creator.Individual(new_alloc)
          archive.append(new_alloc)
        
        #print(new_alloc)
        settings.update({'enable_errors' : False})
        settings.update({'NGEN' : NGEN})
        #run for 10 iterations to find suitable new allocation
        pop, stats, best, archive, archivestats = main(archive = archive, archivestats=archivestats, **settings)
        best = best[0]
        settings.update({'next_alloc' : best})
        settings.update({'enable_errors' : True})
        #run real sim to check next failure
        new_lifetime, new_latency, new_received, new_energy_list, new_node_status, new_missed = evaluate(allocation = new_alloc, repeat = True, **settings)
        actual_times.append(new_lifetime)
        actual_latencies.append(new_latency)
        missed_packages.append(new_missed)
        settings.update({'energy_list_sim' : new_energy_list})
        settings.update({'network_status' : new_node_status})
        #print(f"runtimes so far: {actual_times}")
        #print(f"latencies so far: {actual_latencies}")
        bests.append(list(best))
        bests.append(list(new_alloc))
        front = tools.sortNondominated(pop, len(pop), True)[0]
        fronts.append(front)
        if algorithm  == 'nsga2' or algorithm == 'rmota':
          objectives.append(best.fitness.values)
        else:
          objectives.append((best.fitness.values[0], best.latency))
        run_number +=1
      except exceptions.NetworkDeadException:
        #print(f"Time Elapsed for iteration {i}: {time.time() - start}")
        #old_results = pd.read_sql("results", con=db)
        #min_index = old_results.index.max() + 1 if len(old_results) > 0 else 0
        cleaned_latencies = [x for x in actual_latencies if x < 99999]
        results = {'index' : index,
                     'lifetime' : [sum([x[0] for x in objectives])],
                     'latency' : [max([x[1] for x in objectives])],
                     'actual_lifetime' : [sum(actual_times)],
                     'actual_latency' : [max(cleaned_latencies)] if len(cleaned_latencies) > 0 else 99999,
                     'actual_lifetimes' : json.dumps(actual_times),
                     'actual_latencies' : json.dumps(actual_latencies),
                     'missed_packages' : json.dumps(missed_packages),
                     'algorithm' : algorithm,
                     'settings' : json.dumps(settings),
                     'ntasks': settings['nTasks'],
                     'nnodes' : settings['nNodes'],
                     'task_topology' : settings['task_creator'],
                     'network_topology' : settings['network_creator'],
                     'front' : pickle.dumps(fronts),
                     'bests' : pickle.dumps(bests)}
        df = pd.DataFrame(results, index=[index])
        df.set_index('index', inplace=True)
        #print(df)
        try:
            df.to_sql('results', db, if_exists='append')
        except Exception as e:
            df.to_csv(f"results/{index}.csv")
        print(f"{actual_times}, {actual_latencies}")
        #print(stats)
        break














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
   import sqlalchemy as sql
   import pandas as pd 
   import json
   parser = argparse.ArgumentParser()
   parser.add_argument('-v', action='store_true', default=False, dest='verbose')
   args = parser.parse_args()
   nNodes = 20
   nTasks = 20#for EncodeDecode, this should fit the formula 6x+1 or be 5
   dims =  5#actually, row/column count
   energy = 100
   for algorithm in ['rmota', 'nsga2', 'dtas']:
       crossover = 'nsga2'
       network_creator = 'Grid'
       task_creator = 'OneSink'
       if network_creator == 'Grid':
          nNodes = dims**2
       if task_creator == 'TwoTaskWithProcessing':
          nTasks = 10
       energy_list = [energy]*nNodes
       network_status = [1]*nNodes
          
       settings = {'nNodes' : nNodes,
                 'network_creator' : network_creator,
                 'dimx' : dims,
                 'dimy' : dims,
                 'nTasks' : nTasks,
                 'task_creator' : task_creator,
                 'energy_list_sim' : energy_list,
                 'energy_list_eval': energy_list,
                 'init_energy' : energy,
                 'algorithm' : algorithm,
                 'crossover' : crossover,
                 'verbose' : args.verbose,
                 'capture_packets' : False,
                 'enable_errors' : False,
                 'error_shape' : 1.0,
                 'error_scale' : 10,
                 'network_status' : network_status
                 }

       def min2digits(a):
        s = str(a)
        if len(s) < 2:
           s = "0" + str(a)
        return s
       #ngen for inital pre-run ea
       NGEN = 10
       seed = 2086+42
       offset = 42
       settings = {
                 'experiment' : 'mota',
                 'nNodes' : nNodes,
                 'network_creator' : network_creator,
                 'dimx' : dims,
                 'dimy' : dims,
                 'nTasks' : nTasks,
                 'task_creator' : task_creator,
                 'energy_list_sim' : energy_list,
                 'energy_list_eval': energy_list,
                 'init_energy' : energy,
                 'algorithm' : algorithm,
                 'crossover' : crossover,
                 'verbose' : args.verbose,
                 'capture_packets' : False,
                 'enable_errors' : False,
                 'error_shape' : 1.0,
                 'error_scale' : energy/10,
                 'network_status' : network_status,
                 'NGEN' : NGEN,
                 'run_number' : 1,
                 'next_alloc' : []
                 }
        bests = []
        objectives = []
        recursive = True
        fronts = []
        j = 0
        setup_ea(**settings)
       
        actual_times = []
        actual_latencies = []
        runSeed = seed + i*offset
        settings.update({"prefix" : runSeed})
        pop, stats, best, archive, archivestats = main(seed = runSeed, archive = [], archivestats = {}, **settings)
        best = best[0]
        settings.update({'enable_errors' : True})
        new_lifetime, new_latency, new_received, new_energy_list, new_node_status, new_missed = evaluate(best, **settings)
        actual_times.append(new_lifetime)
        actual_latencies.append(new_latency)
        settings.update({'energy_list_sim' : new_energy_list})
        settings.update({'network_status' : new_node_status})
        #ngen for adjusting
        NGEN = 10
        if settings['algorithm']=='dtas':
           NGEN=25
        settings.update({'NGEN' : NGEN})
        bests.append(list(best))
        front = tools.sortNondominated(pop, len(pop), True)[0]
        fronts.append(front)
        if algorithm  == 'nsga2' or algorithm =='rmota':
          objectives.append(best.fitness.values)
        else:
          objectives.append((best.fitness.values[0], best.latency))
       
        while True:
          try:
            settings.update({'run_number' : j+2})
            try:
              new_alloc = getNewValidAssignment(best, new_node_status, archive, **settings)
            except exceptions.NoValidNodeException:
              raise exceptions.NetworkDeadException

            if len(archive) == 0:
              new_alloc = creator.Individual(new_alloc)
              archive.append(new_alloc)
            #print(new_alloc)
            runSeed = seed + i*offset
            settings.update({"prefix" : runSeed})
            settings.update({'enable_errors' : False})
            settings.update({'NGEN' : NGEN})
            #run for 10 iterations to find suitable new allocation
            pop, stats, best, archive, archivestats = main(seed = runSeed, archive = archive, archivestats=archivestats, **settings)
            best = best[0]
            settings.update({'next_alloc' : best})
            settings.update({'enable_errors' : True})
            #run real sim to check next failure
            new_lifetime, new_latency, new_received, new_energy_list, new_node_status, new_missed = evaluate(allocation = new_alloc, repeat = True, **settings)
            actual_times.append(new_lifetime)
            actual_latencies.append(new_latency)
            settings.update({'energy_list_sim' : new_energy_list})
            settings.update({'network_status' : new_node_status})
            #print(f"runtimes so far: {actual_times}")
            #print(f"latencies so far: {actual_latencies}")
            bests.append(list(best))
            front = tools.sortNondominated(pop, len(pop), True)[0]
            fronts.append(front)
            if algorithm  == 'nsga2' or algorithm == 'rmota':
              objectives.append(best.fitness.values)
            else:
              objectives.append((best.fitness.values[0], best.latency))
             
          except exceptions.NetworkDeadException:
            print(f"Time Elapsed for alg {algorithm}: {time.time() - start}")
            import json
            actual_latencies = [x for x in actual_latencies if x < 99999]
            results = {'index' : min_index,
                         'lifetime' : [sum([x[0] for x in objectives])],
                         'latency' : [max([x[1] for x in objectives])],
                         'actual_lifetime' : [sum(actual_times)],
                         'actual_latency' : [max(actual_latencies)],
                         'settings' : json.dumps(settings),
                         'front' : pickle.dumps(fronts)}
            df = pd.DataFrame(results)
            df.set_index('index', inplace=True)
            print(df)
            print(stats)
            break
