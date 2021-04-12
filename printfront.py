
from __future__ import division
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
from collections import defaultdict, namedtuple
from deap import creator
from deap import tools
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
from individual import ListWithAttributes
import math
import sqlalchemy as sql
import pandas as pd



def min2digits(a):
    s = str(a)
    if len(s) < 2:
      s = "0" + str(a)
    return s

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", ListWithAttributes, fitness=creator.FitnessMin)



parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default = "Grid")
parser.add_argument('--task', type=str, default = 'EncodeDecode')
parser.add_argument('--alg', type=str, default = 'nsga2')
parser.add_argument('--nodes', type=int, default = 81)
parser.add_argument('--tasks', type=int, default = 19)




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


engine = sql.create_engine('postgresql:///dweikert')


df2 = pd.read_sql('results', engine)
print(df2)





def main():
    args = parser.parse_args()


    task = args.task 
    if task == 'Line' or task == 'line':
        task = 'TwoTaskWithProcessing'

    net = args.net
    alg = args.alg
    nNodes = args.nodes
    nTasks = args.tasks

    nNodes = min2digits(nNodes)
    nTasks = min2digits(nTasks)

    best = (-1,-1)
    ref = (0,30000)
    for i in range(31):
        try:
            with open(f"results/{alg}/{net}/Backup/{task}/stats_nodes{nNodes}_tasks{nTasks}_{min2digits(i)}_00.pck", "rb") as f:
                stats = pickle.load(f)
            with open(f"results/{alg}/{net}/Backup/{task}/pop_nodes{nNodes}_tasks{nTasks}_{min2digits(i)}_00.pck", "rb") as f:
                pop = pickle.load(f)
        #print(stats)
        except FileNotFoundError as e:
            break

        pop.sort(key=lambda x: x.fitness.values)
        #print(tools.selNSGA2(pop,1))
        #print(pop[0].fitness.values)
        pop = [x for x in pop if x.fitness.values[1]<99999]
        pop1 = sortEpsilonNondominated(pop, len(pop), True)
        pops = tools.sortNondominated(pop, len(pop), True)
        pop1 = pop1[0]
        pops = pops[0]
        hv = hypervolume(pop)
        if hv > best[0]:
            print(pop1)
            #print(pops[0].fitness.values)
            best = (hv,i)
            front = np.array([ind.fitness.values for ind in pops])
            #plt.scatter(front[:,0], front[:,1]/1000, c="b")
            front = np.array([ind.fitness.values for ind in pop1])
            #plt.scatter(front[:,0], front[:,1]/1000, c="r")
            #plt.axis("tight")
            # plt.show()
        #print(stats[0]['max'])
        print(stats[-1]['min'])
        #print(f"HV : {hypervolume(pop)}")
        #optimal_front = np.array(optimal_front)
        #plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
        #plt.scatter(front[:,0], front[:,1], c="b")
        #plt.axis("tight")
        #plt.show()

    with open(f"results/{alg}/{net}/Backup/{task}/stats_nodes{nNodes}_tasks{nTasks}_{min2digits(best[1])}_00.pck", "rb") as f:
        stats = pickle.load(f)
    with open(f"results/{alg}/{net}/Backup/{task}/pop_nodes{nNodes}_tasks{nTasks}_{min2digits(best[1])}_00.pck", "rb") as f:
        pop = pickle.load(f)
    pop = [x for x in pop if x.fitness.values[1]<99999]
    pop_nondom = tools.sortNondominated(pop, len(pop), True)
    pop_nondom = pop_nondom[0]
    pop_cone = sortEpsilonNondominated(pop, len(pop), True)[0]
    pop_cone = tools.selBest(pop_cone, 1)
    front = np.array([ind.fitness.values for ind in pop_nondom])
    plt.scatter(front[:,0], front[:,1]/1000, c="b")
    front = np.array([ind.fitness.values for ind in pop_cone])
    plt.scatter(front[:,0], front[:,1]/1000, c="r")
    plt.axis("tight")
    ax = plt.gca()
    plt.grid(True)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel("NL (s)")
    plt.ylabel("Latency (s)")
    #plt.title(f"{net}_{nNodes}_{task}_{nTasks}_{min2digits(best[1])}")
    plt.savefig(f"results/plots/{net}_{task}_nodes{nNodes}_tasks{nTasks}_{min2digits(best[1])}_front.png")
    plt.show()
