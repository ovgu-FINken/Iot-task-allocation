
import sqlalchemy as sql
import pandas as pd
import json
import cred
db = sql.create_engine(f'postgresql+psycopg2://{cred.user}:{cred.passwd}@{cred.vm}')
task_setups = ['OneSink']
task_numbers = [20,40,80]
algorithms = ['rmota','nsga2','dtas']

old_results = pd.read_sql("results", con=db)
print(old_results)
def unused():
    network_creator = 'Grid'
    task_creator = ['OneSink']
    nNodes = 81
    dims = 9
    energy = 100
    energy_list = [energy]*nNodes
    network_status = [1]*nNodes
    crossover = 'nsga2'
    NGEN = 100



    for algorithm in algorithms:
        for nTasks in task_numbers:
            for i in range(11):
                settings = {
                         'experiment' : 'rmota',
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
                         'verbose' : False,
                         'capture_packets' : False,
                         'enable_errors' : False,
                         'error_shape' : 1.0,
                         'error_scale' : energy/10,
                         'network_status' : network_status,
                         'NGEN' : NGEN,
                         'run_number' : 1,
                         'next_alloc' : [],
                         }
                old_results = pd.read_sql("results", con=db)
                #min_index = old_results.index.max() + 1 if len(old_results) > 0 else 0
                #min_index = 0 
                #run = {'index' : min_index,
                #   'experiment' : 'rmota',
                #   'algorithm' : algorithm,
                #   'settings' : json.dumps(settings)
                #    }
                #df = pd.DataFrame(run, index=[0])
                #df.set_index('index', inplace=True)
                #df.to_sql('results', db, if_exists='append')
