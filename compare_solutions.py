import pandas as pd
import sqlalchemy as sql
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json
from deap import creator, base
from individual import ListWithAttributes
import pickle as pck
import sys
from exceptions import NetworkDeadException


creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0,))
creator.create("Individual", ListWithAttributes, fitness=creator.FitnessMin)

db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
df =  pd.read_sql('results_surrogates', db)








if __name__ == '__main__':


    i = int(sys.argv[1])
    j = int(sys.argv[2])

    df = df[df['index'] == i]
    df2 = df.iloc[[0]]

    settings = json.loads(df2.iloc[0]['settings'])
    bests = pck.loads(df2.iloc[0]['bests'])
    if j > len(bests):
        print(f"{i} done")
        exit(0)
    if j == len(bests):
        with open(f"surrogate_data/metrics_sim_{i}{j-1}.json", "rb") as f:
            metrics = pck.load(f) 
            db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
            model = settings['eval_mode']
            if model == 'sim':
                model = 'Sim'
            else:
                evalmode = settings['surrogate']
                if evalmode == 'gnn':
                    model = 'GNN'
                else: 
                    model = 'Graph'
            
            results = {'index': i,
                        'lifetime' : metrics[0],
                        'latency' : metrics[1],
                        'missed' : metrics[2],
                        'model' : model,
                        'settings' : json.dumps(settings)
                        }

            df = pd.DataFrame(results, index=[i])
            df.set_index('index', inplace=True)
            df.to_sql('results_surrogates_final', db, if_exists='append')
        

        exit(0)
    


    best = bests[j]
    from network import evaluate
    if j ==0:
        settings['energy_list'] = [100]*settings['nNodes']
        settings['energy_list_sim'] = [100]*settings['nNodes']
        settings['node_status'] = [1]*settings['nNodes']
        lifetime, latency, nMissed, missed_perc, missed_packages, percentage, energy_list = evaluate(list(best), stopTime=20000, **settings) 
        with open(f"surrogate_data/energy_{i}{j+1}.json", "wb+") as f:
            pck.dump(energy_list, f)
        with open(f"surrogate_data/metrics_sim_{i}{j}.json", "wb+") as f:
            pck.dump([lifetime, latency, missed_perc], f)
    if j > 0:
        metrics = []
        energy_list = []
        with open(f"surrogate_data/energy_{i}{j}.json", "rb") as f:
            energy_list = pck.load(f)
        with open(f"surrogate_data/metrics_sim_{i}{j-1}.json", "rb") as f:
            metrics = pck.load(f) 
        settings['energy_list'] = energy_list
        
        try:
            lifetime, latency, nMissed, missed_perc, missed_packages, percentage, energy_list = evaluate(list(best), stopTime=20000, **settings) 
            with open(f"surrogate_data/energy_{i}{j+1}.json", "wb+") as f:
                pck.dump(energy_list, f)
            
            newnl = lifetime + metrics[0]
            
            newl = max(latency, metrics[1])
            if metrics[2] < 1:
                newa = max(missed_perc, metrics[2])
            else:
                newa = missed_perc
            with open(f"surrogate_data/metrics_sim_{i}{j}.json", "wb+") as f:
                pck.dump([newnl, newl, newa], f)
                   
            #print(newnl, newl, newa)

        except Exception as e:
            print(e)
            with open(f"surrogate_data/metrics_sim_{i}{j}.json", "wb+") as f:
                pck.dump([metrics[0], metrics[1], metrics[2]], f)
            print("network dead, aborting")

            exit(0)
        

    #from nsga2 import evaluate_surrogate
    #settings['surrogate'] = 'gnn'
    #lifetime, latency, missed_perc = evaluate_surrogate([best, settings, 0])
    #with open(f"surrogate_data/metrics_gnn_{i}{j}.json", "wb+") as f:
    #    pck.dump([lifetime, latency, missed_perc], f)
    #print(lifetime, latency, missed_perc)
    #
    #settings['surrogate'] = 'graph'
    #lifetime, latency, missed_perc = evaluate_surrogate([best, settings, 0])
    #with open(f"surrogate_data/metrics_gnn_{i}{j}.json", "wb+") as f:
    #    pck.dump([lifetime, latency, missed_perc], f)
    #print(lifetime, latency, missed_perc)
    

