import pandas as pd
import sqlalchemy as sql
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
import json
from deap import creator, base
from individual import ListWithAttributes
import pickle as pck
import random

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0,))
creator.create("Individual", ListWithAttributes, fitness=creator.FitnessMin)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
df =  pd.read_sql('results_final_dmota2', db)

def normalize_abs(series, series2):
    series3 = series.abs()/series2
    return series3.abs()/series3.abs().max()


def get_type(settings):
    if settings['eval_mode'] == 'sim':
        return 'Sim'
    else:
        if 'surrogate' in settings.keys():
            if settings['surrogate'] == 'graph':
                return 'Graph'
            else:
                return 'GNN'
        else: return 'GNN'

def get_type2(settings):
    if settings['eval_mode'] == 'sim':
        return 'Sim'
    else:
        if 'surrogate' in settings.keys():
            if settings['surrogate'] == 'graph':
                return 'Graph'
            else:
                return 'GNN'
        else: return 'GNN'

def get_latency(bests, archives, elitesets):
    lat = 0
    bests = pck.loads(bests)
    for x in bests:
        if x.latency > lat:
            lat = x.latency
    return lat

def get_lifetime(bests, archives, elitesets):
    lifetime = 0
    bests = pck.loads(bests)
    #lifetime += len(bests)
    lifetime += bests[-1].lifetime
    return lifetime

def get_missed(bests, archives, elitesets):
    missed = 0
    bests = pck.loads(bests)
    for x in bests:
        if x.missed_perc > missed:
            missed = x.missed_perc
            #missed += x.missed_perc
    return missed


def fix_alg(alg):
    if alg =='dmota':
        return 'D-MOTA'
    elif alg =='amota':
        return 'A-MOTA'
    elif alg == 'mmota':
        return 'M-MOTA'

df['network'] = df.apply(lambda row: json.loads(row.settings)['network_creator'], axis=1)
df['task'] = df.apply(lambda row: json.loads(row.settings)['task_creator'], axis=1)
df['nTasks'] = df.apply(lambda row: json.loads(row.settings)['nTasks'], axis=1)
df['nNodes'] = df.apply(lambda row: json.loads(row.settings)['nNodes'], axis=1)
df['errorrate'] = df.apply(lambda row: json.loads(row.settings)['error_rate'], axis=1)
df['algorithm'] = df.apply(lambda row: fix_alg(row.algorithm), axis=1)
#df['latency'] = df.apply(lambda row: get_latency(row.bests, row.archives, row.elitesets), axis=1)
#df['lifetime'] = df.apply(lambda row: get_lifetime(row.bests, row.archives, row.elitesets), axis=1)
#df['missed'] = df.apply(lambda row: get_missed(row.bests, row.archives, row.elitesets), axis=1)
#df['type'] = df.apply(lambda row: get_type(json.loads(row.settings)), axis=1)
errorrate = 'high'
df = df[df['nNodes'] > 49]
df = df[df['errorrate'] == errorrate]

df['lifetime'] = normalize_abs(df['lifetime'], df['percentage_missed'])
df = df.sort_values(by='algorithm')
print(df)


#df = df[df['network'] == nw]
#df = df[df['task'] == task]

#df = df.sort_values('type', axis=0)

#dfmota = df[df['type'] == 'Sim']
#dfa = df[df['type'] == 'Graph']
#dfg = df[df['type'] == 'GNN']

#print(dfmota.mean())
#print(dfa.mean())
#print(dfg.mean())
#
#print(dfmota.std())
#print(dfa.std())
#print(dfg.std())
#sns.set_style("whitegrid")
g = sns.catplot(x = 'nnodes', y = 'lifetime', hue='algorithm', data=df, kind='box')
#g.set(yscale='log')
plt.xlabel('Number of nodes')
plt.ylabel('Normalized Network Lifetime')
plt.savefig(f"plots/dmota_{errorrate}_nl.png")
plt.show()
#
g = sns.catplot(x = 'nnodes', y = 'latency', hue='algorithm', data=df, kind='box')
plt.xlabel('Number of nodes')
plt.ylabel('Latency (ms)')
plt.savefig(f"plots/dmota_{errorrate}_l.png")
plt.show()

g = sns.catplot(x = 'nnodes', y = 'percentage_missed', hue='algorithm', data=df, kind='box')
plt.xlabel('Number of nodes')
plt.ylabel('Missed packets (%)')
plt.savefig(f"plots/dmota_{errorrate}_missed.png")
plt.show()
