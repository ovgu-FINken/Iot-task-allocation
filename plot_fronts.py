import json
import pickle
from deap import creator, base
from individual import ListWithAttributes
import math
from collections import defaultdict
import pandas as pd
from individual import ListWithAttributes
from deap import tools
import itertools
import sqlalchemy as sql
import pickle as pck

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0,))
creator.create("Individual", ListWithAttributes, fitness=creator.FitnessMin)

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


db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
df =  pd.read_sql('results_mobility', db)

import matplotlib.pyplot as plt

import numpy as np
df = df[df['algorithm']=='mmota']

colors = ['tab:blue','tab:orange','tab:green', 'tab:olive']*10
i = 0
for index, row in df.iterrows():
    fronts = pck.loads(row['fronts'])
    if len(fronts)>1:
        for x in fronts:
            trimmed_front = [y for y in x if y.fitness.values[1] < 9000]
            if len(trimmed_front) > 3:
                    print(i)
                    i += 1
                    b = sortEpsilonNondominated(trimmed_front, len(trimmed_front), True)[0]
                    f = tools.sortNondominated(trimmed_front,len(trimmed_front), True)[0]
                    if len(b) > 1:
                        b = b[:1]
                    b = np.array([ind.fitness.values for ind in b])
                    f = np.array([ind.fitness.values for ind in f])
                    fig = plt.figure()
                    plt.grid(True)
                    ax = plt.gca()
                    ax.scatter(f[:,0], f[:,1], c = 'b')
                    ax.scatter(b[:,0], b[:,1], c = 'r')
                    plt.xlabel('Network Lifetime (s)')
                    plt.ylabel('Latency (ms)')
                    plt.show()
                    plt.clf()
                    ax = plt.gca()
                    plt.grid(True)
                    ax.scatter(f[:,0], f[:,2], c = 'b')
                    ax.scatter(b[:,0], b[:,2], c = 'r')
                    plt.xlabel('Network Lifetime (s)')
                    plt.ylabel('Number of missed packets')
                    plt.show()






