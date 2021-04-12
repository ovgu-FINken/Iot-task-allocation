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




creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", ListWithAttributes, fitness=creator.FitnessMin)

alg = 'dtas'
alg = 'nsga2'


with open(f"results/{alg}/Grid/EncodeDecode/pop_nodes09_tasks05_00_00.pck", "rb") as f:
   pop = pickle.load(f)


pop.sort(key=lambda x: x.fitness.values)
#print(tools.selNSGA2(pop,1))
#print(pop[0].fitness.values)


print(pop)
print([x.fitness.values for x in pop])
