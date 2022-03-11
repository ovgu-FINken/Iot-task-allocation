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


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
df =  pd.read_sql('results_surrogates_final', db)

random.seed(1002)


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



df['network'] = df.apply(lambda row: json.loads(row.settings)['network_creator'], axis=1)
df['task'] = df.apply(lambda row: json.loads(row.settings)['task_creator'], axis=1)
df['nTasks'] = df.apply(lambda row: json.loads(row.settings)['nTasks'], axis=1)
df['nNodes'] = df.apply(lambda row: json.loads(row.settings)['nNodes'], axis=1)
df['type'] = df.apply(lambda row: get_type(json.loads(row.settings)), axis=1)






df = df[df['network'] == nw]
df = df[df['task'] == task]

df = df.sort_values('type', axis=0)

sns.set_style("whitegrid")
g = sns.catplot(x = 'nNodes', y = 'lifetime', hue='type', data=df, kind='box')
plt.xlabel('Number of nodes')
plt.ylabel('- Network Lifetime (s)')
plt.savefig(f"plots/{nw}_{task}_nl.png")
plt.show()

g = sns.catplot(x = 'nNodes', y = 'latency', hue='type', data=df, kind='box')
plt.xlabel('Number of nodes')
plt.ylabel('Latency (ms)')
plt.savefig(f"plots/{nw}_{task}_l.png")
plt.show()

g = sns.catplot(x = 'nNodes', y = 'missed', hue='type', data=df, kind='box')
plt.xlabel('Number of nodes')
plt.ylabel('Missed packets')
plt.savefig(f"plots/{nw}_{task}_missed.png")
plt.show()
